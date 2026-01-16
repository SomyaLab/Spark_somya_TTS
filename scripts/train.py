"""Training script. Modify config and run: uv run scripts/train.py"""
import sys
sys.path.insert(0, ".")

from src.spark_tts.config import Config
from src.spark_tts.training.trainer import run_training

config = Config(
    data_dir="dataset",
    max_steps=5000,
    batch_size=2,
    learning_rate=2e-4,
    # Resume options:
    # resume_from_checkpoint=True,                      # Auto-detect latest
    # resume_from_checkpoint="outputs/phase1/ckpt-500", # Resume phase 1
    # resume_from_checkpoint="outputs/checkpoint-1000", # Resume phase 2
)

if __name__ == "__main__":
    run_training(config)
