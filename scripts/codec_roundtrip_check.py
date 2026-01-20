"""
Codec round-trip sanity check: audio -> tokens -> audio.

Run:
  uv run python scripts/codec_roundtrip_check.py
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import sys

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spark_tts.config import Config
from src.spark_tts.data.tokenizer import AudioTokenizer

logger = logging.getLogger("spark_tts")


# Defaults (no argparse by request)
DATA_DIR = "datasets"
OUT_DIR = "outputs/codec_roundtrip_check"
NUM_SAMPLES = 55
SAVE_EXAMPLES = 8  # save first N originals + reconstructions
SEED = 42
TARGET_SR = 16000
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 30.0


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    x = torch.from_numpy(audio).float()
    y = torchaudio.functional.resample(x, orig_freq=orig_sr, new_freq=target_sr)
    return y.cpu().numpy().astype(np.float32)


def _parse_ids(token_str: str, pattern: str) -> list[int]:
    return [int(x) for x in re.findall(pattern, token_str)]


def _pad_or_truncate(xs: list[int], n: int, pad_value: int = 0) -> list[int]:
    if len(xs) < n:
        return xs + [pad_value] * (n - len(xs))
    if len(xs) > n:
        return xs[:n]
    return xs


def _si_snr_db(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SNR in dB (robust to amplitude mismatch)."""
    if ref.size == 0 or est.size == 0:
        return float("-inf")
    n = min(ref.size, est.size)
    r = ref[:n].astype(np.float64)
    e = est[:n].astype(np.float64)
    r = r - r.mean()
    e = e - e.mean()
    denom = np.dot(e, e) + eps
    a = float(np.dot(r, e) / denom)
    e_scaled = a * e
    noise = r - e_scaled
    num = float(np.dot(r, r) + eps)
    den = float(np.dot(noise, noise) + eps)
    return 10.0 * np.log10(num / den)


def _stats(wav: np.ndarray) -> dict:
    if wav.size == 0:
        return {"len": 0, "mean_abs": 0.0, "rms": 0.0, "peak": 0.0, "pct_abs_lt_1e3": 100.0}
    wav64 = wav.astype(np.float64)
    mean_abs = float(np.mean(np.abs(wav64)))
    rms = float(np.sqrt(np.mean(np.square(wav64))))
    peak = float(np.max(np.abs(wav64)))
    pct = float(np.mean(np.abs(wav64) < 1e-3) * 100.0)
    return {"len": int(wav.size), "mean_abs": mean_abs, "rms": rms, "peak": peak, "pct_abs_lt_1e3": pct}


def main():
    _setup_logging()
    random.seed(SEED)
    np.random.seed(SEED)

    config = Config()
    logger.info("Config: %s", {k: v for k, v in asdict(config).items() if k in ["model_dir", "device", "sample_rate"]})

    data_dir = Path(DATA_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick WAVs
    wav_paths = sorted(data_dir.rglob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No wavs found under {DATA_DIR}")

    # Filter by duration quickly using soundfile info
    filtered = []
    for p in wav_paths:
        try:
            info = sf.info(str(p))
            if info.frames <= 0:
                continue
            dur = info.frames / float(info.samplerate)
            if MIN_DURATION_SEC <= dur <= MAX_DURATION_SEC:
                filtered.append(p)
        except Exception:
            continue

    if len(filtered) < NUM_SAMPLES:
        logger.warning("Only %d usable wavs after duration filter; using all of them.", len(filtered))
        picked = filtered
    else:
        picked = random.sample(filtered, NUM_SAMPLES)

    logger.info("Picked %d wavs for round-trip check.", len(picked))

    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))
    audio_tokenizer.to(str(config.device))

    REQUIRED_GLOBAL = 32
    snrs = []
    orig_stats = []
    rec_stats = []
    saved = 0

    for i, wav_path in enumerate(picked):
        audio, sr = _load_audio_mono(wav_path)
        audio_16k = _resample(audio, sr, TARGET_SR)

        # Tokenize -> detokenize
        global_str, semantic_str = audio_tokenizer.tokenize_audio(audio, sr)
        global_ids = _parse_ids(global_str, r"<\|bicodec_global_(\d+)\|>")
        semantic_ids = _parse_ids(semantic_str, r"<\|bicodec_semantic_(\d+)\|>")

        global_ids = _pad_or_truncate(global_ids, REQUIRED_GLOBAL, pad_value=0)

        G = torch.tensor(global_ids, dtype=torch.long).unsqueeze(0)  # (1, 32)
        S = torch.tensor(semantic_ids, dtype=torch.long).unsqueeze(0)  # (1, T)

        rec = audio_tokenizer.detokenize(G.to(config.device), S.to(config.device))
        rec = np.asarray(rec, dtype=np.float32)

        snr = _si_snr_db(audio_16k, rec)
        snrs.append(snr)

        o = _stats(audio_16k)
        r = _stats(rec)
        orig_stats.append(o)
        rec_stats.append(r)

        if saved < SAVE_EXAMPLES:
            sf.write(str(out_dir / f"{saved:02d}_orig.wav"), audio_16k, TARGET_SR)
            sf.write(str(out_dir / f"{saved:02d}_recon.wav"), rec, TARGET_SR)
            saved += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(picked):
            logger.info(
                "Progress %d/%d | last_snr=%.2f dB | last_orig_mean_abs=%.4f | last_rec_mean_abs=%.4f",
                i + 1,
                len(picked),
                snr,
                o["mean_abs"],
                r["mean_abs"],
            )

    audio_tokenizer.offload_to_cpu()

    snrs_np = np.array(snrs, dtype=np.float64)
    logger.info("=== Round-trip summary (%d files) ===", len(snrs))
    logger.info("SI-SNR dB | mean=%.2f | median=%.2f | p10=%.2f | p90=%.2f | min=%.2f", float(snrs_np.mean()), float(np.median(snrs_np)), float(np.quantile(snrs_np, 0.1)), float(np.quantile(snrs_np, 0.9)), float(snrs_np.min()))
    logger.info("Saved %d example pairs to %s", saved, out_dir)


if __name__ == "__main__":
    main()

