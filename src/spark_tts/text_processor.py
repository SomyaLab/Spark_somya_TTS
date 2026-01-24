"""
Text Processing for Spark-TTS

Unified text normalization, transliteration, and long text processing for Indic TTS.

Handles:
- Acronym expansion (US -> U S)
- Currency conversion (₹500 -> paanch sau rupaye)
- Symbol replacement (% -> percent)
- Number to words conversion
- English to Indic script transliteration (optional)
- Long text chunking with crossfade for seamless audio
"""

import re
import logging
from typing import Optional
import numpy as np

try:
    from num_to_words import num_to_word
except ImportError:
    num_to_word = None

try:
    from indic_transliteration import sanscript
except ImportError:
    sanscript = None

logger = logging.getLogger("spark_tts")


class IndicLanguageDetector:
    """Detects Indic languages based on Unicode character ranges."""
    
    # Unicode ranges for different scripts
    SCRIPT_RANGES = {
        (0x0000, 0x007F): ["en"],              # ASCII/English
        (0x0900, 0x097F): ["hi", "mr", "sa"],  # Devanagari
        (0x0980, 0x09FF): ["bn"],              # Bengali
        (0x0A80, 0x0AFF): ["gu"],              # Gujarati
        (0x0C00, 0x0C7F): ["te"],              # Telugu
        (0x0C80, 0x0CFF): ["kn"],              # Kannada
    }
    
    # Sanscript scheme mapping
    SCHEME_MAP = {}
    if sanscript:
        SCHEME_MAP = {
            "hi": sanscript.DEVANAGARI,
            "mr": sanscript.DEVANAGARI,
            "sa": sanscript.DEVANAGARI,
            "bn": sanscript.BENGALI,
            "gu": sanscript.GUJARATI,
            "te": sanscript.TELUGU,
            "kn": sanscript.KANNADA,
        }

    @staticmethod
    def detect_script(text: str) -> str:
        """
        Detect the primary script in the text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (e.g., 'hi', 'en', 'kn')
        """
        script_counts = {}
        
        for char in text:
            code_point = ord(char)
            for (start, end), langs in IndicLanguageDetector.SCRIPT_RANGES.items():
                if start <= code_point <= end:
                    lang = langs[0]
                    script_counts[lang] = script_counts.get(lang, 0) + 1
                    break
        
        if not script_counts:
            return "en"
        
        return max(script_counts.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def get_sanscript_scheme(lang_code: str):
        """Get the sanscript scheme for a language code."""
        if not sanscript:
            return None
        return IndicLanguageDetector.SCHEME_MAP.get(
            lang_code, 
            sanscript.DEVANAGARI if sanscript else None
        )


class TextNormalizer:
    """Handles text preprocessing: acronyms, currency, symbols."""
    
    def __init__(self):
        # Matches 2+ uppercase letters (US, NASA, etc.)
        self.acronym_pattern = re.compile(r"\b([A-Z][A-Z0-9&]{1,})\b")
        
        # Matches currency ($50, ₹100, etc.)
        self.currency_pattern = re.compile(r"([$£€₹])\s?(\d+(?:\.\d+)?)")
        
        # Symbol mappings by language
        self.symbols = {
            "en": {
                "%": "percent",
                "&": "and",
                "+": "plus",
                "=": "equals",
                "@": "at",
                "/": "slash"
            },
            "hi": {
                "%": "pratishat",
                "&": "aur",
                "+": "jama",
                "=": "barabar",
                "@": "at",
                "/": "bata"
            }
        }
        
        # Currency names by symbol and language
        self.currency_names = {
            "$": {"en": "dollar", "hi": "dollar"},
            "€": {"en": "euro", "hi": "euro"},
            "£": {"en": "pound", "hi": "pound"},
            "₹": {"en": "rupee", "hi": "rupaye"},
        }

    def _expand_currency(self, match, lang: str = "en") -> str:
        """Convert currency to words."""
        symbol, amount = match.groups()
        
        try:
            if num_to_word:
                val = float(amount) if "." in amount else int(amount)
                num_str = num_to_word(val, lang=lang)
            else:
                num_str = amount
        except Exception:
            num_str = amount
        
        c_lang = lang if lang in self.currency_names["$"] else "en"
        currency_name = self.currency_names.get(symbol, {}).get(c_lang, "currency")
        
        # Pluralize for English
        if c_lang == "en" and float(amount) > 1:
            currency_name += "s"
        
        return f"{num_str} {currency_name}"

    def normalize(self, text: str, lang: str = "en") -> str:
        """
        Apply normalization rules.
        
        1. Expand acronyms (US -> U S)
        2. Convert currency ($5 -> five dollars)
        3. Replace symbols (% -> percent)
        
        Args:
            text: Input text
            lang: Language code for localization
            
        Returns:
            Normalized text
        """
        # 1. Acronyms: Add spaces between letters
        text = self.acronym_pattern.sub(lambda x: " ".join(x.group(0)), text)
        
        # 2. Currency: Convert to words
        text = self.currency_pattern.sub(
            lambda x: self._expand_currency(x, lang), 
            text
        )
        
        # 3. Symbols: Replace with words
        sym_map = self.symbols.get(lang, self.symbols["en"])
        for sym, word in sym_map.items():
            text = text.replace(sym, f" {word} ")
        
        return text


class Transliterator:
    """
    Main text processing class.
    
    Combines normalization with optional transliteration of
    numbers and English words to Indic scripts.
    """
    
    def __init__(self, default_lang: str = "hi"):
        self.default_lang = default_lang
        self.detector = IndicLanguageDetector()
        self.normalizer = TextNormalizer()
        
        # Regex patterns
        self.number_pattern = re.compile(r"-?\d+(?:\.\d+)?")
        self.english_pattern = re.compile(r"\b[a-zA-Z]+\b")

    def transliterate_number(self, number_str: str, lang: str) -> str:
        """Convert number string to words in target language."""
        if not num_to_word:
            return number_str
        try:
            val = float(number_str) if "." in number_str else int(number_str)
            return num_to_word(val, lang=lang)
        except (NotImplementedError, ImportError, Exception):
            return number_str

    def transliterate_english_word(self, word: str, lang: str) -> Optional[str]:
        """Transliterate English word to Indic script."""
        if not sanscript:
            return None
        try:
            target_script = self.detector.get_sanscript_scheme(lang)
            if target_script:
                return sanscript.transliterate(
                    word.lower(), 
                    sanscript.ITRANS, 
                    target_script
                )
        except Exception:
            pass
        return None

    def process_text(
        self,
        text: str,
        lang: Optional[str] = None,
        transliterate_numbers: bool = True,
        transliterate_english: bool = False
    ) -> str:
        """
        Process text with normalization and optional transliteration.
        
        Args:
            text: Input text
            lang: Target language (auto-detected if None)
            transliterate_numbers: Convert numbers to words
            transliterate_english: Transliterate English to Indic script
            
        Returns:
            Processed text
        """
        # 1. Detect language if not provided
        if lang is None:
            lang = self.detector.detect_script(text)
        
        # 2. Apply normalization (acronyms, currency, symbols)
        text = self.normalizer.normalize(text, lang)
        
        # 3. Find and replace numbers/English words
        matches = []
        
        if transliterate_numbers:
            for m in self.number_pattern.finditer(text):
                matches.append({"type": "number", "obj": m})
        
        if transliterate_english:
            for m in self.english_pattern.finditer(text):
                # Skip single letters (from acronym expansion)
                if len(m.group()) > 1:
                    matches.append({"type": "english", "obj": m})
        
        matches.sort(key=lambda x: x["obj"].start())
        
        # 4. Reconstruct text with replacements
        result = []
        last_pos = 0
        
        for match in matches:
            m = match["obj"]
            
            # Add text before match
            if m.start() > last_pos:
                result.append(text[last_pos:m.start()])
            
            content = m.group()
            replacement = content
            
            if match["type"] == "number":
                replacement = self.transliterate_number(content, lang)
            elif match["type"] == "english":
                trans = self.transliterate_english_word(content, lang)
                if trans:
                    replacement = trans
            
            result.append(str(replacement))
            last_pos = m.end()
        
        # Add remaining text
        if last_pos < len(text):
            result.append(text[last_pos:])
        
        # Clean up whitespace
        final_text = "".join(result)
        return re.sub(r"\s+", " ", final_text).strip()


class LongTextProcessor:
    """
    Process long text for TTS by splitting into chunks and generating
    audio with crossfade between chunks.
    """
    
    # Indic sentence boundaries (split points)
    SENTENCE_ENDERS = re.compile(r'([।॥?!.]+)')
    
    # Approximate characters per second for Indic TTS
    # This is a rough estimate: ~15 chars/sec for Indic scripts
    CHARS_PER_SECOND = 15
    
    def __init__(
        self,
        target_duration: float = 20.0,
        sample_rate: int = 16000,
        crossfade_ms: int = 100,
    ):
        """
        Initialize long text processor.
        
        Args:
            target_duration: Target duration per chunk in seconds
            sample_rate: Audio sample rate
            crossfade_ms: Crossfade duration in milliseconds
        """
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.crossfade_ms = crossfade_ms
        self.transliterator = Transliterator()
        
        # Target character count based on duration estimate
        self.target_chars = int(target_duration * self.CHARS_PER_SECOND)
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences at Indic punctuation marks."""
        # Split keeping the delimiters
        parts = self.SENTENCE_ENDERS.split(text)
        
        sentences = []
        current = ""
        
        for i, part in enumerate(parts):
            if self.SENTENCE_ENDERS.match(part):
                # This is a delimiter, append to current sentence
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    def _merge_to_chunks(self, sentences: list[str]) -> list[str]:
        """
        Merge sentences into chunks targeting ~20 seconds each.
        
        Strategy:
        - Merge small sentences until target duration reached
        - Never split a sentence (keep semantic boundaries)
        - Allow some flexibility in chunk size
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds target, start new chunk
            # (unless current chunk is empty)
            if current_length + sentence_len > self.target_chars * 1.2 and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def normalize_and_split(self, text: str, lang: Optional[str] = None) -> list[str]:
        """
        Normalize text and split into chunks for TTS.
        
        1. Normalize text (acronyms, currency, numbers -> words)
        2. Split at sentence boundaries targeting ~20 sec chunks
        
        Args:
            text: Input text
            lang: Target language (auto-detected if None)
            
        Returns:
            List of text chunks ready for TTS
        """
        # Auto-detect language if not provided
        if lang is None:
            lang = IndicLanguageDetector.detect_script(text)
        
        # Normalize entire text first
        normalized = self.transliterator.process_text(text, lang=lang)
        
        # Split into sentences at Indic punctuation
        sentences = self._split_sentences(normalized)
        
        # Merge into chunks targeting ~20 sec each
        chunks = self._merge_to_chunks(sentences)
        
        logger.info(f"Split text into {len(chunks)} chunks (lang={lang})")
        
        return chunks
    
    def crossfade_concat(self, chunks: list[np.ndarray]) -> np.ndarray:
        """
        Concatenate audio chunks with crossfade for smooth transitions.
        
        Args:
            chunks: List of audio arrays (numpy)
            
        Returns:
            Single concatenated audio array
        """
        if not chunks:
            return np.array([], dtype=np.float32)
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Calculate crossfade samples
        crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)
        
        # Filter out empty chunks
        chunks = [c for c in chunks if c is not None and len(c) > 0]
        
        if not chunks:
            return np.array([], dtype=np.float32)
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Calculate total length
        total_length = sum(len(c) for c in chunks) - crossfade_samples * (len(chunks) - 1)
        result = np.zeros(total_length, dtype=np.float32)
        
        # Create crossfade curves
        fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        
        pos = 0
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            
            if i == 0:
                # First chunk: no fade-in, fade-out at end
                if chunk_len > crossfade_samples:
                    result[:chunk_len - crossfade_samples] = chunk[:-crossfade_samples]
                    result[chunk_len - crossfade_samples:chunk_len] = chunk[-crossfade_samples:] * fade_out
                else:
                    result[:chunk_len] = chunk
                pos = chunk_len - crossfade_samples
            elif i == len(chunks) - 1:
                # Last chunk: fade-in at start, no fade-out
                if chunk_len > crossfade_samples:
                    result[pos:pos + crossfade_samples] += chunk[:crossfade_samples] * fade_in
                    result[pos + crossfade_samples:pos + chunk_len] = chunk[crossfade_samples:]
                else:
                    result[pos:pos + chunk_len] += chunk * fade_in[:chunk_len]
            else:
                # Middle chunk: fade-in at start, fade-out at end
                if chunk_len > 2 * crossfade_samples:
                    result[pos:pos + crossfade_samples] += chunk[:crossfade_samples] * fade_in
                    result[pos + crossfade_samples:pos + chunk_len - crossfade_samples] = chunk[crossfade_samples:-crossfade_samples]
                    result[pos + chunk_len - crossfade_samples:pos + chunk_len] = chunk[-crossfade_samples:] * fade_out
                else:
                    # Chunk too short for double crossfade, just blend
                    blend_len = min(chunk_len, crossfade_samples)
                    result[pos:pos + blend_len] += chunk[:blend_len] * fade_in[:blend_len]
                    if chunk_len > blend_len:
                        result[pos + blend_len:pos + chunk_len] = chunk[blend_len:]
                pos += chunk_len - crossfade_samples
        
        return result
    
    def generate_long_audio(
        self,
        text: str,
        ref_audio_path: str,
        model,
        tokenizer,
        audio_tokenizer,
        config,
        lang: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate audio for long text with crossfade between chunks.
        
        Full pipeline: normalize -> chunk -> generate -> crossfade.
        
        Args:
            text: Long input text
            ref_audio_path: Path to reference audio for voice cloning
            model: TTS model
            tokenizer: Text tokenizer
            audio_tokenizer: Audio tokenizer
            config: Model config
            lang: Target language (auto-detected if None)
            
        Returns:
            Single concatenated audio array
        """
        # Import here to avoid circular imports
        from .inference.generate import generate_speech_clone
        
        # Split text into chunks
        chunks = self.normalize_and_split(text, lang)
        
        if not chunks:
            logger.warning("No text chunks to process")
            return np.array([], dtype=np.float32)
        
        # Generate audio for each chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            wav = generate_speech_clone(
                text=chunk,
                ref_audio_path=ref_audio_path,
                model=model,
                tokenizer=tokenizer,
                audio_tokenizer=audio_tokenizer,
                config=config,
            )
            
            if wav is not None and len(wav) > 0:
                audio_chunks.append(wav)
            else:
                logger.warning(f"Chunk {i+1} produced no audio")
        
        if not audio_chunks:
            logger.warning("No audio chunks generated")
            return np.array([], dtype=np.float32)
        
        # Concatenate with crossfade
        result = self.crossfade_concat(audio_chunks)
        
        logger.info(
            f"Generated {len(result) / self.sample_rate:.2f}s audio "
            f"from {len(chunks)} chunks"
        )
        
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

_default_transliterator: Optional[Transliterator] = None
_default_long_processor: Optional[LongTextProcessor] = None


def get_transliterator() -> Transliterator:
    """Get the default transliterator instance."""
    global _default_transliterator
    if _default_transliterator is None:
        _default_transliterator = Transliterator()
    return _default_transliterator


def get_long_text_processor(
    target_duration: float = 20.0,
) -> LongTextProcessor:
    """Get the default long text processor instance."""
    global _default_long_processor
    if _default_long_processor is None:
        _default_long_processor = LongTextProcessor(target_duration=target_duration)
    return _default_long_processor


def normalize_text(text: str, lang: Optional[str] = None) -> str:
    """
    Normalize text for TTS.
    
    Args:
        text: Input text
        lang: Optional language code (auto-detected if None)
        
    Returns:
        Normalized text
    """
    return get_transliterator().process_text(text, lang=lang)


def detect_language(text: str) -> str:
    """
    Detect the primary language/script in text.
    
    Args:
        text: Input text
        
    Returns:
        Language code (e.g., 'hi', 'en', 'kn')
    """
    return IndicLanguageDetector.detect_script(text)
