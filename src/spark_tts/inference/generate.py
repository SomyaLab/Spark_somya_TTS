"""Inference utilities for Spark-TTS (zero-shot voice cloning)."""

import logging
import re
import torch
import numpy as np
import soundfile as sf
from transformers.generation.logits_process import LogitsProcessor
import librosa

from ..config import Config
from ..data.tokenizer import AudioTokenizer
from ..text_processor import normalize_text, IndicLanguageDetector

logger = logging.getLogger("spark_tts")

# Stop tokens for generation (in order of preference)
STOP_TOKENS = ["<|end_semantic_token|>", "<|im_end|>"]

# BiCodec vocab sizes (codebook sizes)
GLOBAL_CODEBOOK_SIZE = 4096
SEMANTIC_CODEBOOK_SIZE = 8192

# During cloning we don't want to stop immediately.
MIN_NEW_SEMANTIC_TOKENS = 200


def _safe_token_id(tokenizer, token: str) -> int | None:
    """Return integer token id, or None if unknown/unconvertible."""
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None:
        return None
    try:
        return int(tid)
    except Exception:
        return None


def _get_stop_token_ids(tokenizer) -> list[int]:
    """Stop token ids, in priority order, skipping unknowns."""
    ids: list[int] = []
    for tok in STOP_TOKENS:
        tid = _safe_token_id(tokenizer, tok)
        if tid is not None:
            ids.append(tid)
    if not ids and tokenizer.eos_token_id is not None:
        ids = [int(tokenizer.eos_token_id)]
    return ids


def _build_codec_token_id_lists(tokenizer) -> tuple[list[int], list[int]]:
    """
    Build (global_token_ids, semantic_token_ids) vocab-id lists.

    Uses a fast contiguous-range check when possible; otherwise falls back to
    per-token lookup (still only a few thousand tokens).
    """
    cached = getattr(tokenizer, "_spark_codec_token_id_lists", None)
    if cached is not None:
        return cached  # type: ignore[return-value]

    def _contiguous_or_fallback(prefix: str, size: int) -> list[int]:
        t0 = _safe_token_id(tokenizer, f"<|{prefix}_0|>")
        t1 = _safe_token_id(tokenizer, f"<|{prefix}_1|>")
        t_last = _safe_token_id(tokenizer, f"<|{prefix}_{size - 1}|>")
        if t0 is not None and t1 is not None and t_last is not None:
            if (t1 == t0 + 1) and (t_last == t0 + (size - 1)):
                return list(range(t0, t0 + size))

        ids: list[int] = []
        for i in range(size):
            tid = _safe_token_id(tokenizer, f"<|{prefix}_{i}|>")
            if tid is None:
                raise ValueError(f"Tokenizer is missing required token: <|{prefix}_{i}|>")
            ids.append(tid)
        return ids

    global_ids = _contiguous_or_fallback("bicodec_global", GLOBAL_CODEBOOK_SIZE)
    semantic_ids = _contiguous_or_fallback("bicodec_semantic", SEMANTIC_CODEBOOK_SIZE)

    tokenizer._spark_codec_token_id_lists = (global_ids, semantic_ids)
    return global_ids, semantic_ids


class _AllowTokenIDsProcessor(LogitsProcessor):
    """Mask logits so ONLY `allowed_token_ids` can be sampled."""

    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = sorted(set(int(x) for x in allowed_token_ids))
        self._mask_cache: dict[tuple[torch.device, int], torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = int(scores.shape[-1])
        key = (scores.device, vocab_size)
        mask = self._mask_cache.get(key)
        if mask is None:
            allowed = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            ids = [i for i in self.allowed_token_ids if 0 <= i < vocab_size]
            if ids:
                allowed[torch.tensor(ids, dtype=torch.long, device=scores.device)] = True
            mask = ~allowed
            self._mask_cache[key] = mask
        return scores.masked_fill(mask, float("-inf"))


def _extract_codec_ids(kind: str, token_strs: list[str]) -> list[int]:
    """Extract BiCodec code ids from token strings like '<|bicodec_{kind}_123|>'."""
    prefix = f"<|bicodec_{kind}_"
    out: list[int] = []
    for t in token_strs:
        if t.startswith(prefix) and t.endswith("|>"):
            # strip "<|bicodec_{kind}_" prefix and "|>" suffix
            out.append(int(t[len(prefix) : -2]))
    return out


def _wav_seems_silent(wav: np.ndarray, eps: float = 1e-4) -> bool:
    if wav is None or wav.size == 0:
        return True
    if not np.isfinite(wav).all():
        return True
    return float(np.mean(np.abs(wav))) < eps


def prepare_wav_for_save(wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Trim silence, normalize, and clip. Use before saving or streaming. Reused by save_audio and server."""
    if wav is None:
        wav = np.array([], dtype=np.float32)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size and not np.isfinite(wav).all():
        logger.warning("Waveform contains NaN/Inf; replacing with zeros")
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    # Trim leading/trailing silence (helps when generation hits max tokens and pads silence).
    if wav.size:
        peak0 = float(np.max(np.abs(wav)))
        trim_eps = max(1e-4, 1e-3 * peak0)
        nz = np.where(np.abs(wav) > trim_eps)[0]
        if nz.size:
            pad = int(0.05 * sample_rate)
            start = max(0, int(nz[0]) - pad)
            end = min(int(wav.shape[0]), int(nz[-1]) + pad + 1)
            if start > 0 or end < int(wav.shape[0]):
                logger.info(
                    "Trimming silence: %.2fs -> %.2fs",
                    float(wav.shape[0]) / float(sample_rate),
                    float(end - start) / float(sample_rate),
                )
                wav = wav[start:end]
    if wav.size:
        peak = float(np.max(np.abs(wav)))
        if peak > 0:
            rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)))
            target_rms = 0.05
            if rms > 0 and rms < target_rms:
                gain = min(target_rms / rms, 30.0)
                wav = wav * gain
            peak = float(np.max(np.abs(wav)))
            if peak > 0 and peak < 0.5:
                wav = wav / peak * 0.95
    if wav.size:
        wav = np.clip(wav, -1.0, 1.0)
    return wav


def save_audio(wav: np.ndarray, path: str, sample_rate: int = 16000):
    """Save waveform to file."""
    wav = prepare_wav_for_save(wav, sample_rate)
    sf.write(path, wav, sample_rate, subtype="PCM_16")
    logger.info("Audio saved to %s (%.2fs)", path, float(wav.shape[0]) / float(sample_rate) if wav.size else 0.0)


@torch.inference_mode()
def generate_speech_clone(
    text: str,
    ref_audio_path: str,
    model,
    tokenizer,
    audio_tokenizer: AudioTokenizer,
    config: Config,
) -> np.ndarray:
    """
    Zero-shot voice cloning: generate speech in a reference speaker's voice.

    Extracts global tokens from reference audio and generates semantic tokens
    conditioned on those globals.

    Args:
        text: Input text to convert to speech
        ref_audio_path: Path to reference audio file for voice cloning
        model: Trained model
        tokenizer: Text tokenizer
        audio_tokenizer: Audio tokenizer for tokenization/detokenization
        config: Configuration with inference settings

    Returns:
        Generated waveform as numpy array
    """ 
    
    torch.compiler.reset()

    # Normalize text (acronyms, currency, numbers -> words)
    lang = IndicLanguageDetector.detect_script(text)
    text = normalize_text(text, lang=lang)
    logger.info(f"Normalized text (lang={lang}): {text[:100]}...")

    # Load and tokenize reference audio to extract speaker identity
    ref_audio, ref_sr = librosa.load(ref_audio_path, sr=None)
    ref_global_tokens, _ = audio_tokenizer.tokenize_audio(ref_audio, ref_sr)
    
    logger.info(f"Extracted global tokens from reference: {ref_audio_path}")

    # Build prompt with global tokens already included (voice identity injected)
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>",
        ref_global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
    ])

    # Constrained semantic decoding (codec-only + stop tokens)
    global_vocab_ids, semantic_vocab_ids = _build_codec_token_id_lists(tokenizer)
    stop_token_ids = _get_stop_token_ids(tokenizer)

    rep_pen = getattr(config, "repetition_penalty", None)
    gen_kwargs = {}
    if rep_pen is not None:
        gen_kwargs["repetition_penalty"] = float(rep_pen)

    model_inputs = tokenizer([prompt], return_tensors="pt").to(config.device)
    semantic_processor = _AllowTokenIDsProcessor(semantic_vocab_ids + stop_token_ids)
    out = model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_audio_tokens,
        # Greedy decoding is the most stable starting point for semantic tokens.
        do_sample=True,
        min_new_tokens=int(getattr(config, "min_new_semantic_tokens", MIN_NEW_SEMANTIC_TOKENS)),
        eos_token_id=stop_token_ids,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=[semantic_processor],
        **gen_kwargs,
    )
    gen_ids = out[:, model_inputs.input_ids.shape[1] :]
    gen_tokens = tokenizer.convert_ids_to_tokens(gen_ids[0].tolist(), skip_special_tokens=False)
    semantic_ids = _extract_codec_ids("semantic", gen_tokens)

    if not semantic_ids:
        logger.warning("No semantic tokens found with constrained decoding.")
        return np.array([], dtype=np.float32)

    pred_semantic_ids = torch.tensor(semantic_ids).long().unsqueeze(0)

    # Extract global tokens from reference (already have them as string, need IDs)
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", ref_global_tokens)
    pred_global_ids = torch.tensor(
        [int(t) for t in global_matches]
    ).long().unsqueeze(0).unsqueeze(0)

    logger.info(f"Semantic tokens: {pred_semantic_ids.shape[1]}, Global tokens: {pred_global_ids.shape[2]}")

    # Detokenize to audio
    audio_tokenizer.to(str(config.device))
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(config.device).squeeze(0),
        pred_semantic_ids.to(config.device),
    )

    return wav_np
