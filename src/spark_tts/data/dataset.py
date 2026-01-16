"""Dataset loading utilities for Spark-TTS."""

import json
import hashlib
import logging
import random
from pathlib import Path
from collections import defaultdict

import librosa
import soundfile as sf
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from .tokenizer import AudioTokenizer

logger = logging.getLogger("spark_tts")

# Default duration limits (can be overridden via config)
DEFAULT_MIN_DURATION = 0.5   # seconds
DEFAULT_MAX_DURATION = 20.0  # seconds

# Marker for completion start (loss computed from here)
COMPLETION_START_TOKEN = "<|start_global_token|>"

# Sequence length filtering constants
# Conservative character-to-token ratio estimate (most tokens are 1-4 chars)
CHAR_TO_TOKEN_RATIO = 4


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds without loading full audio."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        # Fallback: load with librosa
        audio, sr = librosa.load(audio_path, sr=None)
        return len(audio) / sr


def preprocess_sample(
    audio_path: str,
    text: str,
    audio_tokenizer: AudioTokenizer,
) -> dict[str, str]:
    """
    Preprocess a single audio sample for training.
    Returns dict with 'prompt' and 'completion' for loss masking.
    
    Note: Speaker identity is encoded in global tokens, not text prefix.
    This ensures train/inference consistency for zero-shot cloning.
    
    Args:
        audio_path: Path to audio file
        text: Transcript text
        audio_tokenizer: AudioTokenizer instance
        
    Returns:
        Dict with 'prompt' and 'completion' keys
        
    Raises:
        ValueError: If audio loading or tokenization fails
    """
    if not text or not text.strip():
        raise ValueError(f"Empty text for audio: {audio_path}")
    
    try:
        audio_array, sr = librosa.load(audio_path, sr=None)
        if len(audio_array) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to load audio {audio_path}: {e}") from e
    
    try:
        global_tokens, semantic_tokens = audio_tokenizer.tokenize_audio(audio_array, sr)
        if not global_tokens or not semantic_tokens:
            raise ValueError(f"Empty tokens from audio: {audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to tokenize audio {audio_path}: {e}") from e

    # Prompt: text content (loss will be masked)
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
    ])
    
    # Completion: audio tokens (loss will be computed)
    completion = "".join([
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|end_semantic_token|>",
        "<|im_end|>",
    ])

    return {"prompt": prompt, "completion": completion}


def preprocess_cloning_pair(
    ref_audio_path: str,
    target_audio_path: str,
    target_text: str,
    audio_tokenizer: AudioTokenizer,
) -> dict[str, str]:
    """
    Preprocess a cross-utterance cloning pair for zero-shot training.
    
    Uses global tokens from ref_audio (speaker A) and text + semantic tokens 
    from target_audio (same speaker, different utterance B).
    
    This teaches the model to use global tokens as speaker identity 
    independent of the actual utterance content.
    
    Args:
        ref_audio_path: Path to reference audio (for speaker identity)
        target_audio_path: Path to target audio (for content)
        target_text: Text transcript for target audio
        audio_tokenizer: AudioTokenizer instance
        
    Returns:
        Dict with 'prompt' and 'completion' keys
        
    Raises:
        ValueError: If audio loading or tokenization fails
    """
    if not target_text or not target_text.strip():
        raise ValueError(f"Empty text for target audio: {target_audio_path}")
    
    # Extract global tokens from reference audio (speaker identity)
    try:
        ref_audio, ref_sr = librosa.load(ref_audio_path, sr=None)
        if len(ref_audio) == 0:
            raise ValueError(f"Empty reference audio: {ref_audio_path}")
        ref_global_tokens, _ = audio_tokenizer.tokenize_audio(ref_audio, ref_sr)
        if not ref_global_tokens:
            raise ValueError(f"Empty global tokens from reference: {ref_audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to process reference audio {ref_audio_path}: {e}") from e
    
    # Extract semantic tokens from target audio (content)
    try:
        target_audio, target_sr = librosa.load(target_audio_path, sr=None)
        if len(target_audio) == 0:
            raise ValueError(f"Empty target audio: {target_audio_path}")
        _, target_semantic_tokens = audio_tokenizer.tokenize_audio(target_audio, target_sr)
        if not target_semantic_tokens:
            raise ValueError(f"Empty semantic tokens from target: {target_audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to process target audio {target_audio_path}: {e}") from e
    
    # Prompt: text content (loss will be masked)
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        target_text,
        "<|end_content|>",
    ])
    
    # Completion: ref's global + target's semantic (loss will be computed)
    completion = "".join([
        "<|start_global_token|>",
        ref_global_tokens,  # From reference (speaker identity)
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        target_semantic_tokens,  # From target (content)
        "<|end_semantic_token|>",
        "<|im_end|>",
    ])
    
    return {"prompt": prompt, "completion": completion}


def build_cloning_pairs(
    samples: list[dict],
    audio_tokenizer: AudioTokenizer,
    max_seq_length: int = 2048,
    pairs_per_speaker: int = 5,
) -> list[dict]:
    """
    Build cross-utterance cloning pairs from samples grouped by speaker.
    
    For each speaker with multiple utterances, creates pairs where:
    - Global tokens come from one utterance (reference)
    - Text + semantic tokens come from another utterance (target)
    
    Args:
        samples: List of sample dicts with audio_path, text, speaker
        audio_tokenizer: AudioTokenizer instance
        max_seq_length: Maximum sequence length for filtering
        pairs_per_speaker: Number of cloning pairs to generate per speaker
        
    Returns:
        List of processed cloning pair dicts
    """
    # Group samples by speaker
    by_speaker = defaultdict(list)
    for sample in samples:
        by_speaker[sample["speaker"]].append(sample)
    
    # Filter to speakers with at least 2 utterances
    multi_utterance_speakers = {
        speaker: utts for speaker, utts in by_speaker.items() 
        if len(utts) >= 2
    }
    
    if not multi_utterance_speakers:
        logger.warning("No speakers with multiple utterances for cloning pairs")
        return []
    
    logger.info(f"Building cloning pairs from {len(multi_utterance_speakers)} speakers")
    
    cloning_samples = []
    for speaker, utterances in tqdm(multi_utterance_speakers.items(), desc="Building cloning pairs"):
        # Generate random pairs for this speaker
        num_pairs = min(pairs_per_speaker, len(utterances) * (len(utterances) - 1) // 2)
        
        for _ in range(num_pairs):
            # Pick two different utterances
            ref, target = random.sample(utterances, 2)
            
            try:
                tokenized = preprocess_cloning_pair(
                    ref_audio_path=ref["audio_path"],
                    target_audio_path=target["audio_path"],
                    target_text=target["text"],
                    audio_tokenizer=audio_tokenizer,
                )
                
                # Filter by length (prompt + completion) - rough character estimate
                total_len = len(tokenized["prompt"]) + len(tokenized["completion"])
                if total_len > max_seq_length * CHAR_TO_TOKEN_RATIO:
                    continue
                    
                cloning_samples.append(tokenized)
            except Exception as e:
                logger.warning(f"Skipping cloning pair {ref['audio_path']} -> {target['audio_path']}: {e}")
    
    logger.info(f"Generated {len(cloning_samples)} cloning pairs")
    return cloning_samples


def tokenize_for_training(
    samples: list[dict],
    tokenizer,
    max_seq_length: int = 2048,
) -> list[dict]:
    """
    Tokenize samples and create labels with -100 for prompt tokens.
    
    This enables completion-only loss: loss is only computed on audio tokens,
    not on the text prompt.
    
    Args:
        samples: List of dicts with 'prompt' and 'completion' keys
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length (will truncate if exceeded)
        
    Returns:
        List of dicts with 'input_ids', 'attention_mask', 'labels'
    """
    tokenized_samples = []
    skipped_count = 0
    
    for sample in tqdm(samples, desc="Tokenizing for training", unit="sample"):
        try:
            prompt = sample["prompt"]
            completion = sample["completion"]
            
            # Tokenize prompt and completion separately
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            
            # Combine
            input_ids = prompt_ids + completion_ids
            prompt_len = len(prompt_ids)
            
            # Truncate if needed (preserve as much prompt as possible, truncate from completion)
            if len(input_ids) > max_seq_length:
                # If prompt is longer than max_seq_length, truncate prompt too
                if prompt_len >= max_seq_length:
                    # Entire sequence is prompt (edge case - should be filtered earlier)
                    prompt_len = max_seq_length
                    input_ids = input_ids[:max_seq_length]
                else:
                    # Truncate from completion side, keep full prompt
                    input_ids = input_ids[:max_seq_length]
                    # prompt_len stays the same
            
            # Ensure labels match input_ids length exactly
            # Create labels: -100 for prompt tokens (no loss), actual IDs for completion tokens
            labels = [-100] * prompt_len + input_ids[prompt_len:].copy()
            
            # Verify lengths match
            assert len(labels) == len(input_ids), \
                f"Label length {len(labels)} != input_ids length {len(input_ids)}"
            
            # Attention mask (all 1s for now, padding handled by collator)
            attention_mask = [1] * len(input_ids)
            
            tokenized_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        except Exception as e:
            skipped_count += 1
            logger.warning(f"Error tokenizing sample: {e}")
            continue
    
    if skipped_count > 0:
        logger.warning(f"⚠️  Skipped {skipped_count} samples during tokenization")
    
    return tokenized_samples


def get_wav_files(speaker_dir: Path) -> list[Path]:
    """Get all WAV files from speaker's wav/ subdirectory."""
    wav_dir = speaker_dir / "wav"
    if wav_dir.exists():
        return sorted(wav_dir.glob("*.wav"))
    return []


def load_transcriptions(data_dir: Path) -> dict[str, dict]:
    """Load transcriptions from the IISc SYSPIN format."""
    transcripts = {}
    for json_file in data_dir.glob("*_Transcripts.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "Transcripts" in data:
                for key, value in data["Transcripts"].items():
                    transcripts[key] = {
                        "transcript": value.get("Transcript", ""),
                        "domain": value.get("Domain", ""),
                    }
    return transcripts


def load_local_dataset(
    data_dir: str,
    audio_tokenizer: AudioTokenizer,
    tokenizer,
    speaker_name: str | None = None,
    cache_dir: str = ".cache/tokenized_dataset",
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_seq_length: int = 2048,
    model_dir: str | None = None,
    sample_rate: int = 16000,
    use_cloning_pairs: bool = True,
    cloning_pairs_per_speaker: int = 5,
) -> Dataset:
    """
    Load local dataset from IISc SYSPIN data structure.
    Returns pre-tokenized HuggingFace Dataset with input_ids, attention_mask, labels.
    Labels have -100 for prompt tokens (loss masked) and actual IDs for completion.
    
    Args:
        data_dir: Path to dataset directory
        audio_tokenizer: AudioTokenizer instance for audio-to-tokens
        tokenizer: HuggingFace tokenizer for text tokenization
        speaker_name: Optional filter for specific speaker
        cache_dir: Directory for caching tokenized dataset
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        max_seq_length: Maximum sequence length (for post-tokenization filter)
        model_dir: Model directory (for cache key)
        sample_rate: Audio sample rate (for cache key)
        use_cloning_pairs: Whether to generate cross-utterance cloning pairs
        cloning_pairs_per_speaker: Number of cloning pairs per speaker
    """
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    
    # Include all params in cache key for proper invalidation
    cache_str = f"{data_dir}:{speaker_name}:{model_dir}:{min_duration}:{max_duration}:{max_seq_length}:{CHAR_TO_TOKEN_RATIO}:{sample_rate}:{use_cloning_pairs}:{cloning_pairs_per_speaker}"
    cache_key = hashlib.md5(cache_str.encode()).hexdigest()[:12]
    dataset_cache = cache_path / cache_key
    
    if dataset_cache.exists():
        logger.info(f"Loading cached dataset from {dataset_cache}")
        cached_dataset = load_from_disk(str(dataset_cache))
        logger.info(f"Cached dataset size: {len(cached_dataset)} samples")
        if len(cached_dataset) < 100:
            logger.warning(f"⚠️  Cached dataset seems unusually small ({len(cached_dataset)} samples)")
            logger.warning("   Consider clearing cache if this is unexpected")
        return cached_dataset
    
    samples = []
    duration_filtered = 0

    # Find all speaker directories
    speaker_dirs = []
    for lang_dir in data_path.iterdir():
        if not lang_dir.is_dir():
            continue
        syspin_data = lang_dir / "IISc_SYSPIN_Data"
        if not syspin_data.exists():
            continue
        for d in syspin_data.iterdir():
            if d.is_dir() and d.name.startswith("IISc_SYSPINProject_"):
                if speaker_name is None or speaker_name in d.name:
                    speaker_dirs.append(d)

    for speaker_dir in speaker_dirs:
        transcripts = load_transcriptions(speaker_dir)
        wav_files = get_wav_files(speaker_dir)

        for wav_path in wav_files:
            audio_id = wav_path.stem
            if audio_id not in transcripts:
                continue

            transcript_data = transcripts[audio_id]
            text = transcript_data["transcript"]
            if not text.strip():
                continue

            # Duration filter (before expensive tokenization)
            duration = get_audio_duration(str(wav_path))
            if duration < min_duration or duration > max_duration:
                duration_filtered += 1
                continue

            samples.append({
                "audio_path": str(wav_path),
                "text": text,
                "speaker": speaker_dir.name,
            })

    if duration_filtered > 0:
        logger.info(f"Filtered {duration_filtered} samples by duration ({min_duration}-{max_duration}s)")

    if not samples:
        raise ValueError(f"No valid samples found in {data_dir}")

    # Process regular samples (same-utterance: text + global + semantic from same audio)
    processed = []
    seq_length_filtered = 0
    for sample in tqdm(samples, desc="Tokenizing audio", unit="file"):
        try:
            tokenized = preprocess_sample(
                audio_path=sample["audio_path"],
                text=sample["text"],
                audio_tokenizer=audio_tokenizer,
            )
            # Quick character-based filter (rough estimate to avoid expensive tokenization)
            # Most tokens are 1-4 chars, so max_seq_length * CHAR_TO_TOKEN_RATIO is conservative
            total_len = len(tokenized["prompt"]) + len(tokenized["completion"])
            if total_len > max_seq_length * CHAR_TO_TOKEN_RATIO:
                seq_length_filtered += 1
                continue
            processed.append(tokenized)
        except Exception as e:
            logger.warning(f"Skipping {sample['audio_path']}: {e}")
            continue

    if seq_length_filtered > 0:
        logger.info(f"Filtered {seq_length_filtered} samples exceeding max_seq_length")
    
    logger.info(f"Processed {len(processed)} regular samples")

    # Build cross-utterance cloning pairs for zero-shot training
    if use_cloning_pairs:
        cloning_samples = build_cloning_pairs(
            samples=samples,
            audio_tokenizer=audio_tokenizer,
            max_seq_length=max_seq_length,
            pairs_per_speaker=cloning_pairs_per_speaker,
        )
        processed.extend(cloning_samples)
        logger.info(f"Total samples (regular + cloning): {len(processed)}")

    # Tokenize and create labels with -100 for prompt tokens (completion-only loss)
    logger.info("Creating tokenized dataset with loss masking...")
    tokenized_samples = tokenize_for_training(processed, tokenizer, max_seq_length)
    
    # Filter out samples that are still too long after tokenization (shouldn't happen often)
    final_samples = []
    final_filtered = 0
    for sample in tokenized_samples:
        if len(sample["input_ids"]) <= max_seq_length:
            final_samples.append(sample)
        else:
            final_filtered += 1
    
    if final_filtered > 0:
        logger.warning(f"Filtered {final_filtered} samples that exceeded max_seq_length after tokenization")
    
    if not final_samples:
        raise ValueError("No valid samples after tokenization and filtering")
    
    dataset = Dataset.from_list(final_samples)
    logger.info(f"Final dataset size: {len(dataset)} samples")
    
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dataset_cache))
    logger.info(f"Cached dataset to {dataset_cache}")

    return dataset
