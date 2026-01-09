import re
import torch
import numpy as np
from unsloth import FastModel

from ..config import Config
from ..data.tokenizer import AudioTokenizer


@torch.inference_mode()
def generate_speech(
    text: str,
    model,
    tokenizer,
    audio_tokenizer: AudioTokenizer,
    config: Config,
    speaker: str | None = None,
) -> np.ndarray:
    """
    Generate speech audio from text.

    Args:
        text: Input text to convert to speech
        model: Trained model
        tokenizer: Text tokenizer
        audio_tokenizer: Audio tokenizer for detokenization
        config: Configuration with inference settings
        speaker: Optional speaker identifier for multi-speaker

    Returns:
        Generated waveform as numpy array
    """
    torch.compiler.reset()

    # Prepare prompt
    content = f"{speaker}: {text}" if speaker else text
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        content,
        "<|end_content|>",
        "<|start_global_token|>",
    ])

    model_inputs = tokenizer([prompt], return_tensors="pt").to(config.device)

    # Generate tokens
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_audio_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Trim input tokens
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

    # Extract semantic tokens
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", generated_text)
    if not semantic_matches:
        print("Warning: No semantic tokens found in generated output")
        return np.array([], dtype=np.float32)

    pred_semantic_ids = torch.tensor(
        [int(t) for t in semantic_matches]
    ).long().unsqueeze(0)

    # Extract global tokens
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", generated_text)
    if not global_matches:
        print("Warning: No global tokens found, using zeros")
        pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
    else:
        pred_global_ids = torch.tensor(
            [int(t) for t in global_matches]
        ).long().unsqueeze(0)

    pred_global_ids = pred_global_ids.unsqueeze(0)

    print(f"Semantic tokens: {pred_semantic_ids.shape[1]}")
    print(f"Global tokens: {pred_global_ids.shape[2]}")

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
    print(f"Audio saved to {path}")
