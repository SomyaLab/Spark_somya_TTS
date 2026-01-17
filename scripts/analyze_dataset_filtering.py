"""
Analyze Spark-TTS dataset filtering on the real dataset (NOT the cached dataset).

How it works:
- Uses the same Config + model/tokenizer + AudioTokenizer as training
- Forces a unique cache_dir each run so it can't reuse an old cache
- Prints stage-by-stage counts from the dataset loader logs

Run (recommended):
  /teamspace/studios/this_studio/.venv/bin/python scripts/analyze_dataset_filtering.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works when running from scripts/
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.spark_tts.config import Config
from src.spark_tts.training.trainer import setup_logging, load_model
from src.spark_tts.data.tokenizer import AudioTokenizer
from src.spark_tts.data.dataset import load_local_dataset


logger = logging.getLogger("spark_tts")


def analyze() -> None:
    config = Config()
    setup_logging()

    # Force a fresh, unique cache dir every run so we don't accidentally load old caches.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = Path(".cache") / "tokenized_dataset_analysis" / run_id

    logger.info("=" * 80)
    logger.info("DATASET FILTERING ANALYSIS (fresh run, no cache reuse)")
    logger.info("=" * 80)
    logger.info(f"data_dir: {config.data_dir}")
    logger.info(f"cache_dir: {cache_dir}")
    logger.info(f"min_duration: {config.min_audio_duration}s")
    logger.info(f"max_duration: {config.max_audio_duration}s")
    logger.info(f"max_seq_length: {config.max_seq_length}")
    logger.info(f"use_cloning_pairs: {config.use_cloning_pairs}")
    logger.info("")

    logger.info("Loading model + tokenizer (same as training)...")
    model, tokenizer = load_model(config)

    logger.info("Loading dataset (this will print detailed filtering logs + tqdm progress)...")
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))
    dataset = load_local_dataset(
        data_dir=config.data_dir,
        audio_tokenizer=audio_tokenizer,
        tokenizer=tokenizer,
        cache_dir=str(cache_dir),
        min_duration=config.min_audio_duration,
        max_duration=config.max_audio_duration,
        max_seq_length=config.max_seq_length,
        model_dir=config.model_dir,
        sample_rate=config.sample_rate,
        use_cloning_pairs=config.use_cloning_pairs,
        # cloning_pairs_per_speaker uses dataset.py default unless you change it there
    )
    audio_tokenizer.offload_to_cpu()

    logger.info("=" * 80)
    logger.info(f"FINAL: {len(dataset)} samples")
    logger.info("=" * 80)


if __name__ == "__main__":
    analyze()

