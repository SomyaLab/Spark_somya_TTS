"""
Normalize local dataset folders into the canonical layout expected by
`src/spark_tts/data/dataset.py::load_local_dataset()`.

Goal (single root):
  datasets/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_*/wav/*.wav
  datasets/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_*/*_Transcripts.json

What this script does (idempotent):
- Flatten IISc SYSPIN "double nested" structure:
    datasets/<lang>/IISc_SYSPIN_Data/IISc_SYSPIN_Data/IISc_SYSPINProject_*  -> move up one level
- Include IISc SPICOR by moving + renaming (as requested):
    datasets/<lang>/IISc_SPICOR_Data/IISc_SPICOR_Data/IISc_SPICORProject_*
      -> datasets/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_SPICOR_*

No argparse by request: edit constants below if needed.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


# =========================
# User-tunable defaults
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = REPO_ROOT / "datasets"

# If True, run a fast structural validation at the end
VALIDATE = True

# If True, stop at first serious error
FAIL_FAST = False


@dataclass
class Stats:
    syspin_speakers_moved: int = 0
    syspin_speakers_skipped_exists: int = 0
    spicor_speakers_moved: int = 0
    spicor_speakers_skipped_exists: int = 0
    removed_empty_dirs: int = 0


def _is_lang_dir(p: Path) -> bool:
    return p.is_dir() and 2 <= len(p.name) <= 4 and p.name.islower()


def _remove_if_empty(p: Path, stats: Stats) -> None:
    try:
        if p.exists() and p.is_dir() and not any(p.iterdir()):
            p.rmdir()
            stats.removed_empty_dirs += 1
    except Exception:
        # best effort only
        pass


def _move_dir(src: Path, dst: Path, *, stats: Stats, exists_counter_attr: str, moved_counter_attr: str) -> None:
    """
    Move a whole directory if destination doesn't exist; otherwise skip.
    We keep it conservative to avoid accidental overwrites.
    """
    if dst.exists():
        setattr(stats, exists_counter_attr, getattr(stats, exists_counter_attr) + 1)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    setattr(stats, moved_counter_attr, getattr(stats, moved_counter_attr) + 1)


def _flatten_syspin(lang_dir: Path, stats: Stats) -> None:
    outer = lang_dir / "IISc_SYSPIN_Data"
    inner = outer / "IISc_SYSPIN_Data"
    if not inner.exists() or not inner.is_dir():
        return

    for speaker_dir in sorted(inner.iterdir()):
        if not speaker_dir.is_dir():
            continue
        if not speaker_dir.name.startswith("IISc_SYSPINProject_"):
            continue
        dst = outer / speaker_dir.name
        _move_dir(
            speaker_dir,
            dst,
            stats=stats,
            exists_counter_attr="syspin_speakers_skipped_exists",
            moved_counter_attr="syspin_speakers_moved",
        )

    _remove_if_empty(inner, stats)


def _include_spicor(lang_dir: Path, stats: Stats) -> None:
    spicor_outer = lang_dir / "IISc_SPICOR_Data"
    spicor_inner = spicor_outer / "IISc_SPICOR_Data"
    if not spicor_inner.exists() or not spicor_inner.is_dir():
        return

    syspin_outer = lang_dir / "IISc_SYSPIN_Data"
    syspin_outer.mkdir(parents=True, exist_ok=True)

    for spk_dir in sorted(spicor_inner.iterdir()):
        if not spk_dir.is_dir():
            continue
        if not spk_dir.name.startswith("IISc_SPICORProject_"):
            continue

        new_name = "IISc_SYSPINProject_SPICOR_" + spk_dir.name.removeprefix("IISc_SPICORProject_")
        dst = syspin_outer / new_name
        _move_dir(
            spk_dir,
            dst,
            stats=stats,
            exists_counter_attr="spicor_speakers_skipped_exists",
            moved_counter_attr="spicor_speakers_moved",
        )

    _remove_if_empty(spicor_inner, stats)
    _remove_if_empty(spicor_outer, stats)


def _validate_layout(datasets_root: Path) -> None:
    """
    Validate alignment with `load_local_dataset` expectations:
    - Finds speaker dirs under: datasets/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_*
    - Checks transcript ids vs wav stems for each speaker.
    """
    print("\nValidating canonical layout (fast structural check)...")

    lang_dirs = sorted([p for p in datasets_root.iterdir() if _is_lang_dir(p)])
    speaker_dirs: list[Path] = []
    for lang in lang_dirs:
        syspin = lang / "IISc_SYSPIN_Data"
        if not syspin.exists():
            continue
        for d in syspin.iterdir():
            if d.is_dir() and d.name.startswith("IISc_SYSPINProject_"):
                speaker_dirs.append(d)

    missing_wavs_total = 0
    missing_transcripts_total = 0
    speakers_with_issues = 0

    for spk in tqdm(sorted(speaker_dirs), desc="validate", unit="speaker", dynamic_ncols=True):
        wav_dir = spk / "wav"
        wav_stems = set(w.stem for w in wav_dir.glob("*.wav")) if wav_dir.exists() else set()

        transcript_ids: set[str] = set()
        for tf in spk.glob("*_Transcripts.json"):
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k in (data.get("Transcripts") or {}).keys():
                    transcript_ids.add(str(k))
            except Exception as e:
                print(f"\n[WARN] failed reading {tf}: {e}")

        missing_wavs = transcript_ids - wav_stems
        missing_transcripts = wav_stems - transcript_ids
        if missing_wavs or missing_transcripts:
            speakers_with_issues += 1
            missing_wavs_total += len(missing_wavs)
            missing_transcripts_total += len(missing_transcripts)
            print(f"\n[WARN] {spk}")
            if missing_wavs:
                print(f"  transcript_ids_missing_wav: {len(missing_wavs)} (e.g. {list(sorted(missing_wavs))[:3]})")
            if missing_transcripts:
                print(f"  wavs_missing_transcript:   {len(missing_transcripts)} (e.g. {list(sorted(missing_transcripts))[:3]})")

    print("\nValidation summary:")
    print(f"  speaker_dirs_checked:        {len(speaker_dirs)}")
    print(f"  speakers_with_issues:        {speakers_with_issues}")
    print(f"  transcript_ids_missing_wav:  {missing_wavs_total}")
    print(f"  wavs_missing_transcript:     {missing_transcripts_total}")
    if speakers_with_issues == 0:
        print("  ✅ Compatible with `load_local_dataset` folder scan assumptions.")
    else:
        print("  ⚠ Fix issues above (common cause: interrupted download before transcript shard flush).")


def main() -> None:
    if not DATASETS_ROOT.exists():
        raise FileNotFoundError(f"datasets root not found: {DATASETS_ROOT}")

    stats = Stats()
    lang_dirs = sorted([p for p in DATASETS_ROOT.iterdir() if _is_lang_dir(p)])

    print(f"datasets_root: {DATASETS_ROOT}")
    print(f"languages: {', '.join([p.name for p in lang_dirs])}")

    for lang_dir in tqdm(lang_dirs, desc="normalize", unit="lang", dynamic_ncols=True):
        try:
            _flatten_syspin(lang_dir, stats)
            _include_spicor(lang_dir, stats)
        except Exception as e:
            msg = f"[ERROR] {lang_dir}: {e}"
            print(msg)
            if FAIL_FAST:
                raise

    print("\nNormalization summary:")
    print(f"  syspin_speakers_moved:          {stats.syspin_speakers_moved}")
    print(f"  syspin_speakers_skipped_exists: {stats.syspin_speakers_skipped_exists}")
    print(f"  spicor_speakers_moved:          {stats.spicor_speakers_moved}")
    print(f"  spicor_speakers_skipped_exists: {stats.spicor_speakers_skipped_exists}")
    print(f"  removed_empty_dirs:             {stats.removed_empty_dirs}")

    if VALIDATE:
        _validate_layout(DATASETS_ROOT)


if __name__ == "__main__":
    main()

