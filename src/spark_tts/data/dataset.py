import os
import json
from pathlib import Path

import librosa
import numpy as np
from datasets import Dataset

from .tokenizer import AudioTokenizer


def get_wav_files(data_dir: Path) -> list[Path]:
    """Get all WAV files from a directory."""
    return sorted(data_dir.glob("*.wav"))


def load_transcriptions(data_dir: Path) -> dict[str, dict]:
    """
    Load transcriptions from the IISc SYSPIN format.

    Returns dict mapping audio ID to {transcript, domain}.
    """
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
    speaker_name: str | None = None,
) -> Dataset:
    """
    Load local dataset from IISc SYSPIN data structure.

    Args:
        data_dir: Root data directory containing speaker folders
        audio_tokenizer: AudioTokenizer instance for processing
        speaker_name: Optional specific speaker folder to use

    Returns:
        HuggingFace Dataset with tokenized text field
    """
    data_path = Path(data_dir)
    samples = []

    # Find all speaker directories
    speaker_dirs = []
    for d in data_path.iterdir():
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

            samples.append({
                "audio_path": str(wav_path),
                "text": text,
                "speaker": speaker_dir.name,
            })

    if not samples:
        raise ValueError(f"No valid samples found in {data_dir}")

    # Process samples
    processed = []
    for sample in samples:
        try:
            tokenized = preprocess_sample(
                audio_path=sample["audio_path"],
                text=sample["text"],
                audio_tokenizer=audio_tokenizer,
                speaker=sample.get("speaker"),
            )
            processed.append(tokenized)
        except Exception as e:
            print(f"Skipping {sample['audio_path']}: {e}")
            continue

    return Dataset.from_list(processed)


def preprocess_sample(
    audio_path: str,
    text: str,
    audio_tokenizer: AudioTokenizer,
    speaker: str | None = None,
) -> dict[str, str]:
    """
    Preprocess a single audio sample for training.

    Args:
        audio_path: Path to audio file
        text: Transcript text
        audio_tokenizer: AudioTokenizer instance
        speaker: Optional speaker identifier

    Returns:
        Dict with 'text' field containing formatted tokens
    """
    # Load audio
    audio_array, sr = librosa.load(audio_path, sr=None)

    # Tokenize audio
    global_tokens, semantic_tokens = audio_tokenizer.tokenize_audio(audio_array, sr)

    # Format for training
    content = f"{speaker}: {text}" if speaker else text

    formatted = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        content,
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|end_semantic_token|>",
        "<|im_end|>",
    ])

    return {"text": formatted}
