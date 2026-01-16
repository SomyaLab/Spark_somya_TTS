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

    # Get correct EOS token for stopping generation
    eos_token_id = _get_eos_token_id(tokenizer)

    # Generate tokens
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_audio_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Trim input tokens
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

    logger.debug(f"Generated output (first 500 chars): {generated_text[:500]}")

    # Extract semantic tokens
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
    if not semantic_matches:
        logger.warning("No semantic tokens found in generated output")
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


def save_audio(wav: np.ndarray, path: str, sample_rate: int = 16000):
    """Save waveform to file."""
    import soundfile as sf
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

    # Get correct EOS token for stopping generation
    eos_token_id = _get_eos_token_id(tokenizer)

    # Generate only semantic tokens (global tokens already provided)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_audio_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Trim input tokens
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

    logger.debug(f"Generated output (first 500 chars): {generated_text[:500]}")

    # Extract semantic tokens from generated output
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
    if not semantic_matches:
        logger.warning("No semantic tokens found in generated output")
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
