"""
Quick training script with customizable config.
Modify the config below and run: uv run scripts/train.py
"""

import sys
sys.path.insert(0, ".")

from src.spark_tts.config import Config
from src.spark_tts.data.tokenizer import AudioTokenizer
from src.spark_tts.data.dataset import load_local_dataset
from src.spark_tts.training.trainer import load_model, setup_lora, train, save_model


# Customize training here
config = Config(
    data_dir="data/IISc_SYSPIN_Data",
    max_steps=60,
    batch_size=2,
    learning_rate=2e-4,
)


if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model(config)

    print("Setting up LoRA...")
    model = setup_lora(model, config)

    print("Initializing audio tokenizer...")
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))

    print(f"Loading dataset from {config.data_dir}...")
    dataset = load_local_dataset(config.data_dir, audio_tokenizer)
    print(f"Dataset size: {len(dataset)} samples")

    audio_tokenizer.offload_to_cpu()

    print("Starting training...")
    train(model, tokenizer, dataset, config)

    print("Saving model...")
    save_model(model, tokenizer, config)

    print("Done!")
