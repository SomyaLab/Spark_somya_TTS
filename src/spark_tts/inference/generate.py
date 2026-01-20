"""Inference utilities for Spark-TTS."""

import logging
import re
import torch
import numpy as np
from unsloth import FastModel

from ..config import Config
from ..data.tokenizer import AudioTokenizer

logger = logging.getLogger("spark_tts")

# Stop tokens for generation (in order of preference)
STOP_TOKENS = ["<|end_semantic_token|>", "<|im_end|>"]


def _get_eos_token_id(tokenizer) -> int:
    """Get the appropriate EOS token ID for generation."""
    for token in STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            return token_id
    # Fallback to default EOS
    return tokenizer.eos_token_id


@torch.inference_mode()
def _generate_text(
    model,
    model_inputs,
    tokenizer,
    config: Config,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float | None = None,
) -> str:
    eos_token_id = _get_eos_token_id(tokenizer)
    gen_kwargs = {}
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_audio_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **gen_kwargs,
    )
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]


@torch.inference_mode()
def generate_speech(
    text: str,
    model,
    tokenizer,
    audio_tokenizer: AudioTokenizer,
    config: Config,
) -> np.ndarray:
    """
    Generate speech audio from text.

    Args:
        text: Input text to convert to speech
        model: Trained model
        tokenizer: Text tokenizer
        audio_tokenizer: Audio tokenizer for detokenization
        config: Configuration with inference settings

    Returns:
        Generated waveform as numpy array
    """
    torch.compiler.reset()

    # Prepare prompt (no speaker prefix - identity comes from global tokens)
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>",
    ])

    model_inputs = tokenizer([prompt], return_tensors="pt").to(config.device)

    generated_text = _generate_text(
        model,
        model_inputs,
        tokenizer,
        config,
        do_sample=True,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=getattr(config, "repetition_penalty", None),
    )

    logger.info(f"Generated output (first 300 chars): {generated_text[:300]}")

    # Extract semantic tokens
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
    if not semantic_matches:
        logger.warning("No semantic tokens found (sampling). Retrying with greedy decoding.")
        generated_text = _generate_text(
            model,
            model_inputs,
            tokenizer,
            config,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=getattr(config, "repetition_penalty", None),
        )
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
        if not semantic_matches:
            logger.warning("No semantic tokens found in generated output (even after greedy retry)")
            return np.array([], dtype=np.float32)

    pred_semantic_ids = torch.tensor(
        [int(t) for t in semantic_matches]
    ).long().unsqueeze(0)

    # Extract global tokens (need exactly 32 for speaker encoder)
    REQUIRED_GLOBAL_TOKENS = 32
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", generated_text)
    if not global_matches:
        logger.warning("No global tokens found, padding with zeros")
        pred_global_ids = torch.zeros((1, REQUIRED_GLOBAL_TOKENS), dtype=torch.long)
    else:
        global_ids = [int(t) for t in global_matches]
        if len(global_ids) < REQUIRED_GLOBAL_TOKENS:
            logger.warning(f"Only {len(global_ids)} global tokens, padding to {REQUIRED_GLOBAL_TOKENS}")
            global_ids = global_ids + [0] * (REQUIRED_GLOBAL_TOKENS - len(global_ids))
        elif len(global_ids) > REQUIRED_GLOBAL_TOKENS:
            global_ids = global_ids[:REQUIRED_GLOBAL_TOKENS]
        pred_global_ids = torch.tensor(global_ids).long().unsqueeze(0)

    pred_global_ids = pred_global_ids.unsqueeze(0)

    logger.info(f"Semantic tokens: {pred_semantic_ids.shape[1]}, Global tokens: {pred_global_ids.shape[2]}")

    # Detokenize to audio
    audio_tokenizer.to(str(config.device))
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(config.device).squeeze(0),
        pred_semantic_ids.to(config.device),
    )

    return wav_np


def _wav_seems_silent(wav: np.ndarray, eps: float = 1e-4) -> bool:
    if wav is None or wav.size == 0:
        return True
    if not np.isfinite(wav).all():
        return True
    return float(np.mean(np.abs(wav))) < eps


def save_audio(wav: np.ndarray, path: str, sample_rate: int = 16000):
    """Save waveform to file."""
    import soundfile as sf
    if wav is None:
        wav = np.array([], dtype=np.float32)
    # Ensure sane dtype/range for players.
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size and not np.isfinite(wav).all():
        logger.warning("Waveform contains NaN/Inf; replacing with zeros")
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize if extremely quiet (common when detokenizer outputs low amplitude).
    if wav.size:
        peak = float(np.max(np.abs(wav)))
        if peak > 0:
            # RMS normalization helps when output is "technically non-zero" but perceived as silence.
            rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)))
            target_rms = 0.05
            if rms > 0 and rms < target_rms:
                gain = min(target_rms / rms, 30.0)  # cap amplification
                wav = wav * gain
            # Also ensure we have a healthy peak.
            peak = float(np.max(np.abs(wav)))
            if peak > 0 and peak < 0.5:
                wav = wav / peak * 0.95
    # Soft clip to [-1, 1] to avoid overflow in PCM conversion.
    if wav.size:
        wav = np.clip(wav, -1.0, 1.0)
    sf.write(path, wav, sample_rate)
    logger.info(f"Audio saved to {path}")


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
    import librosa
    
    torch.compiler.reset()

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

    model_inputs = tokenizer([prompt], return_tensors="pt").to(config.device)

    generated_text = _generate_text(
        model,
        model_inputs,
        tokenizer,
        config,
        do_sample=True,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )

    logger.info(f"Generated output (first 300 chars): {generated_text[:300]}")

    # Extract semantic tokens from generated output
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
    if not semantic_matches:
        logger.warning("No semantic tokens found (sampling). Retrying with greedy decoding.")
        generated_text = _generate_text(
            model,
            model_inputs,
            tokenizer,
            config,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=config.repetition_penalty,
        )
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
        if not semantic_matches:
            logger.warning("No semantic tokens found in generated output (even after greedy retry)")
            return np.array([], dtype=np.float32)

    pred_semantic_ids = torch.tensor(
        [int(t) for t in semantic_matches]
    ).long().unsqueeze(0)

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
