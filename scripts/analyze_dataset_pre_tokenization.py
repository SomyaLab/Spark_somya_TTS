"""
Pre-tokenization dataset analysis (fast, no model/tokenizer/audio-tokenizer, no cache).

Mirrors the *pre-tokenization* stages inside `load_local_dataset()`:
1) discover speaker dirs
2) match wav files with transcripts + non-empty text
3) duration filter (min/max seconds)

Run:
  /teamspace/studios/this_studio/.venv/bin/python scripts/analyze_dataset_pre_tokenization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm import tqdm

# Ensure repo root is on sys.path so `import src...` works when running from scripts/
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.spark_tts.config import Config


def _load_transcriptions(speaker_dir: Path) -> dict[str, str]:
    """Return {audio_id: transcript_text} for this speaker dir."""
    transcripts: dict[str, str] = {}
    for json_file in speaker_dir.glob("*_Transcripts.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = data.get("Transcripts") if isinstance(data, dict) else None
        if not isinstance(t, dict):
            continue
        for key, value in t.items():
            if not isinstance(value, dict):
                continue
            transcripts[str(key)] = str(value.get("Transcript", "") or "")
    return transcripts


def _get_wav_files(speaker_dir: Path) -> list[Path]:
    wav_dir = speaker_dir / "wav"
    return sorted(wav_dir.glob("*.wav")) if wav_dir.exists() else []


def _duration_seconds(path: str) -> float:
    """Fast duration via soundfile; fallback to librosa if needed."""
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:
        try:
            import librosa

            y, sr = librosa.load(path, sr=None)
            return float(len(y) / sr) if sr else 0.0
        except Exception:
            return 0.0


def main() -> None:
    cfg = Config()
    data_path = Path(cfg.data_dir)
    if not data_path.exists():
        raise SystemExit(f"Data directory not found: {data_path}")

    print("=" * 80)
    print("PRE-TOKENIZATION DATASET ANALYSIS")
    print("=" * 80)
    print(f"data_dir: {data_path}")
    print(f"duration_filter: {cfg.min_audio_duration}s .. {cfg.max_audio_duration}s")
    print("")

    # 1) Discover speaker dirs
    speaker_dirs: list[Path] = []
    for lang_dir in sorted([p for p in data_path.iterdir() if p.is_dir()]):
        syspin_data = lang_dir / "IISc_SYSPIN_Data"
        if not syspin_data.exists():
            continue
        for d in syspin_data.iterdir():
            if d.is_dir() and d.name.startswith("IISc_SYSPINProject_"):
                speaker_dirs.append(d)

    print(f"Speaker directories: {len(speaker_dirs)}")

    # 2) Match wavs with transcripts + non-empty text
    total_wavs = 0
    matched_text = 0
    no_transcript = 0
    empty_text = 0

    matched_paths: list[str] = []

    for speaker_dir in tqdm(speaker_dirs, desc="Scanning speakers", unit="speaker"):
        transcripts = _load_transcriptions(speaker_dir)
        wav_files = _get_wav_files(speaker_dir)
        total_wavs += len(wav_files)

        for wav_path in wav_files:
            audio_id = wav_path.stem
            text = transcripts.get(audio_id)
            if text is None:
                no_transcript += 1
                continue
            if not text.strip():
                empty_text += 1
                continue
            matched_text += 1
            matched_paths.append(str(wav_path))

    print("")
    print("-" * 80)
    print("STAGE 2: transcript matching")
    print("-" * 80)
    print(f"Total WAV files found:            {total_wavs}")
    print(f"Matched (has transcript + text):  {matched_text}")
    print(f"Missing transcript:               {no_transcript}")
    print(f"Empty transcript text:            {empty_text}")

    # 3) Duration filter (only for matched, mirrors loader)
    too_short = 0
    too_long = 0
    duration_errors = 0
    kept = 0

    print("")
    print("-" * 80)
    print("STAGE 3: duration filtering (on matched samples)")
    print("-" * 80)

    for p in tqdm(matched_paths, desc="Checking durations", unit="wav"):
        d = _duration_seconds(p)
        if d <= 0:
            duration_errors += 1
            continue
        if d < cfg.min_audio_duration:
            too_short += 1
            continue
        if d > cfg.max_audio_duration:
            too_long += 1
            continue
        kept += 1

    print(f"Kept after duration filter:        {kept}")
    print(f"Too short (<{cfg.min_audio_duration}s):          {too_short}")
    print(f"Too long (>{cfg.max_audio_duration}s):           {too_long}")
    print(f"Duration read errors:              {duration_errors}")

    print("")
    print("=" * 80)
    print("SUMMARY (pre-tokenization)")
    print("=" * 80)
    if total_wavs:
        print(f"raw_wavs: {total_wavs} (100.0%)")
    if total_wavs:
        print(f"matched_text: {matched_text} ({matched_text / total_wavs * 100:.1f}%)")
    if matched_text:
        print(f"kept_after_duration: {kept} ({kept / matched_text * 100:.1f}% of matched)")
    if total_wavs:
        print(f"overall_kept_after_duration: {kept} ({kept / total_wavs * 100:.1f}% of raw)")
    print("=" * 80)


if __name__ == "__main__":
    main()

