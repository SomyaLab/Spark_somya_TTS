"""
Spark-TTS: Text-to-Speech training and inference.

Usage:
    # Training
    python main.py train

    # Inference
    python main.py infer "Your text here"
"""

import sys
import torch
from unsloth import FastModel

from src.spark_tts.config import Config
from src.spark_tts.data.tokenizer import AudioTokenizer
from src.spark_tts.data.dataset import load_local_dataset
from src.spark_tts.training.trainer import load_model, setup_lora, train, save_model
from src.spark_tts.inference.generate import generate_speech, save_audio


def run_training(config: Config):
    """Run the training pipeline."""
    print("Loading model...")
    model, tokenizer = load_model(config)

    print("Setting up LoRA...")
    model = setup_lora(model, config)

    print("Initializing audio tokenizer...")
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))

    print(f"Loading dataset from {config.data_dir}...")
    dataset = load_local_dataset(config.data_dir, audio_tokenizer)
    print(f"Dataset size: {len(dataset)} samples")

    # Offload audio models to save GPU memory
    audio_tokenizer.offload_to_cpu()

    print("Starting training...")
    metrics = train(model, tokenizer, dataset, config)

    print("Saving model...")
    save_model(model, tokenizer, config)

    return metrics


def run_inference(text: str, config: Config, speaker: str | None = None):
    """Run inference to generate speech."""
    print("Loading model...")
    model, tokenizer = load_model(config)

    # Load LoRA weights if available
    try:
        model.load_adapter(config.lora_dir)
        print(f"Loaded LoRA weights from {config.lora_dir}")
    except Exception:
        print("No LoRA weights found, using base model")

    FastModel.for_inference(model)

    print("Initializing audio tokenizer...")
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))

    print(f"Generating speech for: '{text}'")
    wav = generate_speech(
        text=text,
        model=model,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        config=config,
        speaker=speaker,
    )

    if wav.size > 0:
        output_path = "generated_speech.wav"
        save_audio(wav, output_path, config.sample_rate)
        return output_path
    else:
        print("Generation failed - no tokens produced")
        return None


def main():
    config = Config()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "train":
        run_training(config)

    elif command == "infer":
        if len(sys.argv) < 3:
            print("Usage: python main.py infer \"Your text here\"")
            return
        text = sys.argv[2]
        speaker = sys.argv[3] if len(sys.argv) > 3 else None
        run_inference(text, config, speaker)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
