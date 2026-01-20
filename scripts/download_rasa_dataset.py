"""
Download + restructure the HuggingFace dataset `ai4bharat/Rasa` into the local
IISc SYSPIN-style layout expected by `src/spark_tts/data/dataset.py`.

Expected output layout (per speaker):
  dataset/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_*/wav/*.wav
  dataset/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_*/*_Transcripts.json

Notes:
- `ai4bharat/Rasa` is gated on the Hub → you must be authenticated.
  - Option A: set env var `HF_TOKEN` before running
  - Option B: run `huggingface-cli login` once

No argparse by request: edit the constants below if needed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset, Audio
from tqdm import tqdm

try:
    from huggingface_hub import login as hf_login
except Exception:  # pragma: no cover
    hf_login = None

try:
    from datasets import get_dataset_config_names
except Exception:  # pragma: no cover
    get_dataset_config_names = None


# =========================
# User-tunable defaults
# =========================

DATASET_ID = "ai4bharat/Rasa"

# Merge into the repo's existing dataset tree (this repo already has `datasets/`).
# Training scripts default to `dataset/`, so we also create/refresh a `dataset` symlink
# pointing at `datasets` for convenience.
OUTPUT_DATA_DIRNAME = "datasets"

# If empty, auto-detect language codes you already "have" from existing folders under
# repo_root/dataset/* and/or repo_root/datasets/*.
LANGUAGE_CODES_TO_INCLUDE: list[str] = []

SPLITS_TO_TRY = ["train", "validation", "test"]

# Prefer non-streaming for stability.
# In this environment, streaming iteration can sometimes crash the interpreter at shutdown,
# even when `Audio(decode=False)` is used.
USE_STREAMING = False

# Keep memory bounded: shard transcript JSON files per speaker.
TRANSCRIPTS_PER_JSON_SHARD = 5_000

# Limit for quick experiments; set to None for full download.
MAX_UTTERANCES_TOTAL: int | None = None

# Prefer hardlinks from HF cache when possible (falls back to copy).
HARDLINK_FROM_CACHE_IF_POSSIBLE = True

# After download, validate that every transcript id has a matching wav filename
# and that every wav has a transcript entry (matches `load_local_dataset` expectations).
VERIFY_OUTPUT_AFTER_DOWNLOAD = True


# =========================
# Language mapping helpers
# =========================

# Common Indian language names → repo codes (extend if needed).
LANG_NAME_TO_CODE: dict[str, str] = {
    "Gujarati": "gu",
    "Bengali": "bn",
    "Bhojpuri": "bh",
    "Chhattisgarhi": "hne",
    "Hindi": "hi",
    "Kannada": "kn",
    "Magahi": "mag",
    "Maithili": "mai",
    "Marathi": "mr",
    "Telugu": "te",
    # Also present in Rasa configs (not currently used unless you add the code to your repo):
    # "Assamese": "as",
    # "Tamil": "ta",
    # ...
}
LANG_CODE_TO_NAME: dict[str, str] = {v: k for k, v in LANG_NAME_TO_CODE.items()}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_name(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_\\-]+", "", s)
    return s or "unknown"


def _normalize_gender(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"m", "male"}:
        return "Male"
    if s in {"f", "female"}:
        return "Female"
    return None


def _gender_abbrev(gender: str) -> str:
    return "m" if gender == "Male" else "f"


def _normalize_language_to_code(value: Any) -> str | None:
    """
    Rasa can expose language either as a language name (e.g., "Hindi")
    or as a short code (e.g., "hi") depending on the dataset builder.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    low = raw.lower()
    if re.fullmatch(r"[a-z]{2,4}", low):
        return low

    titled = re.sub(r"\s+", " ", raw)
    titled = titled[:1].upper() + titled[1:]
    return LANG_NAME_TO_CODE.get(titled)


def _detect_lang_codes_you_have(repo_root: Path) -> list[str]:
    codes: set[str] = set()
    for base in [repo_root / "dataset", repo_root / "datasets"]:
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir() and re.fullmatch(r"[a-z]{2,4}", child.name):
                codes.add(child.name)
    return sorted(codes)


def _try_hf_login_from_env() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return
    if hf_login is None:
        print("Warning: huggingface_hub not available; cannot auto-login with HF_TOKEN.")
        return
    hf_login(token=token, add_to_git_credential=False)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if HARDLINK_FROM_CACHE_IF_POSSIBLE:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def _export_audio(sample: dict[str, Any], dst_wav: Path) -> None:
    """
    Prefer using a cached on-disk wav path if available; else write from decoded arrays.
    """
    wav_path = sample.get("wav_path")
    if wav_path:
        src = Path(str(wav_path))
        if src.exists():
            _link_or_copy(src, dst_wav)
            return

    audio = sample.get("audio")
    if isinstance(audio, dict):
        p = audio.get("path")
        if p:
            src = Path(str(p))
            if src.exists():
                _link_or_copy(src, dst_wav)
                return

        # If we cast Audio(decode=False), datasets may return raw file bytes.
        b = audio.get("bytes")
        if b:
            dst_wav.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_wav, "wb") as f:
                f.write(b if isinstance(b, (bytes, bytearray)) else bytes(b))
            return

        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is not None and sr is not None:
            import soundfile as sf

            dst_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst_wav, arr, int(sr))
            return

    raise RuntimeError(f"Could not locate audio for sample keys={list(sample.keys())}")


@dataclass
class _SpeakerShardState:
    speaker_dir: Path
    lang_code: str
    gender: str
    utterance_idx: int = 0
    part_idx: int = 0
    buffer: dict[str, dict[str, str]] | None = None  # audio_id -> {Transcript, Domain}

    def __post_init__(self) -> None:
        self.buffer = {}

    def next_audio_id(self, split: str, base: str | None) -> str:
        self.utterance_idx += 1
        suffix = f"_{_safe_name(base)[:40]}" if base else ""
        return _safe_name(
            f"RASA_{self.lang_code}_{_gender_abbrev(self.gender)}_{split}_{self.utterance_idx:09d}{suffix}"
        )

    def add(self, audio_id: str, transcript: str, domain: str) -> None:
        assert self.buffer is not None
        self.buffer[audio_id] = {"Transcript": transcript, "Domain": domain}

    def flush_if_needed(self, force: bool = False) -> int:
        assert self.buffer is not None
        if not force and len(self.buffer) < TRANSCRIPTS_PER_JSON_SHARD:
            return 0
        if not self.buffer:
            return 0

        self.part_idx += 1
        out_name = f"RASA_{self.lang_code}_{_gender_abbrev(self.gender)}_part{self.part_idx:04d}_Transcripts.json"
        out_path = self.speaker_dir / out_name

        payload = {
            "MetaData": {
                "Title": "Rasa Transcripts (converted to IISc SYSPIN format)",
                "Source": DATASET_ID,
            },
            "SpeakersMetaData": {
                "Language": LANG_CODE_TO_NAME.get(self.lang_code, self.lang_code),
                "Gender": self.gender,
            },
            "Transcripts": self.buffer,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

        n = len(self.buffer)
        self.buffer.clear()
        return n


def _validate_syspin_layout(root: Path, lang_codes: list[str]) -> None:
    """
    Fast structural validation for `load_local_dataset` compatibility:
    - speaker_dir must contain wav/*.wav
    - speaker_dir must contain one or more *_Transcripts.json
    - transcript keys must match wav stems
    """
    print("\nValidating on-disk dataset structure (fast check, no audio decoding)...")
    total_speakers = 0
    total_transcript_ids = 0
    total_wavs = 0
    total_missing_wavs = 0
    total_missing_transcripts = 0

    speaker_dirs: list[Path] = []
    for code in lang_codes:
        lang_dir = root / code
        syspin_data = lang_dir / "IISc_SYSPIN_Data"
        if not syspin_data.exists():
            continue
        for d in syspin_data.iterdir():
            if d.is_dir() and d.name.startswith("IISc_SYSPINProject_"):
                speaker_dirs.append(d)

    for spk_dir in tqdm(sorted(speaker_dirs), desc="validate", unit="speaker", dynamic_ncols=True):
        total_speakers += 1
        wav_dir = spk_dir / "wav"
        wav_stems = set()
        if wav_dir.exists():
            for w in wav_dir.glob("*.wav"):
                wav_stems.add(w.stem)

        transcript_files = list(spk_dir.glob("*_Transcripts.json"))
        transcripts: set[str] = set()
        for tf in transcript_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k in (data.get("Transcripts") or {}).keys():
                    transcripts.add(str(k))
            except Exception as e:
                print(f"Warning: failed reading {tf}: {e}")

        missing_wavs = transcripts - wav_stems
        missing_transcripts = wav_stems - transcripts

        total_transcript_ids += len(transcripts)
        total_wavs += len(wav_stems)
        total_missing_wavs += len(missing_wavs)
        total_missing_transcripts += len(missing_transcripts)

        # Print a few examples if something is wrong
        if missing_wavs or missing_transcripts:
            print(f"\n[WARN] {spk_dir}")
            if missing_wavs:
                print(f"  transcript_ids_missing_wav: {len(missing_wavs)} (e.g. {list(sorted(missing_wavs))[:3]})")
            if missing_transcripts:
                print(f"  wavs_missing_transcript:   {len(missing_transcripts)} (e.g. {list(sorted(missing_transcripts))[:3]})")

    print("\nValidation summary:")
    print(f"  speakers:                 {total_speakers}")
    print(f"  transcript_ids_total:     {total_transcript_ids}")
    print(f"  wav_files_total:          {total_wavs}")
    print(f"  transcript_ids_missing_wav: {total_missing_wavs}")
    print(f"  wavs_missing_transcript:    {total_missing_transcripts}")
    if total_missing_wavs == 0 and total_missing_transcripts == 0:
        print("  ✅ Layout is compatible with `load_local_dataset` naming assumptions.")
    else:
        print("  ⚠ Some mismatches found (see warnings above).")


def main() -> None:
    repo_root = _repo_root()
    out_root = repo_root / OUTPUT_DATA_DIRNAME
    out_root.mkdir(parents=True, exist_ok=True)

    # IMPORTANT SAFETY NOTE:
    # `dataset/` may be a symlink to `datasets/` (we create it below). If so, both
    # paths point to the same directory, and any "merge legacy dataset -> datasets"
    # logic can accidentally delete data. So we only attempt any merge if `dataset/`
    # is a *real directory* AND it resolves to a different location than `datasets/`.
    legacy_root = repo_root / "dataset"
    try:
        legacy_is_safe_dir = (
            out_root.name == "datasets"
            and legacy_root.exists()
            and legacy_root.is_dir()
            and not legacy_root.is_symlink()
            and legacy_root.resolve() != out_root.resolve()
        )
    except Exception:
        legacy_is_safe_dir = False

    if legacy_is_safe_dir:
        print(f"Found legacy dataset dir at {legacy_root} (merging into {out_root})")
        for child in legacy_root.iterdir():
            if not child.is_dir():
                continue
            target = out_root / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
            else:
                # Only move files that don't already exist. Never delete existing targets.
                for sub in child.rglob("*"):
                    if sub.is_dir():
                        continue
                    rel = sub.relative_to(child)
                    dst = target / rel
                    if dst.exists():
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(sub), str(dst))
        # Do not delete legacy_root automatically; leave it for manual cleanup.

    # Keep training defaults working: ensure `dataset` points to `datasets`.
    if out_root.name == "datasets":
        link_path = repo_root / "dataset"
        try:
            if link_path.is_symlink() or link_path.exists():
                # If it's a real directory, we already attempted to merge above. Remove if empty.
                if link_path.is_symlink():
                    pass
                elif link_path.is_dir() and not any(link_path.iterdir()):
                    link_path.rmdir()
            if not link_path.exists():
                os.symlink(out_root, link_path)
        except Exception as e:
            print(f"Warning: could not create `dataset` symlink -> `datasets`: {e}")

    lang_codes = [c.strip().lower() for c in LANGUAGE_CODES_TO_INCLUDE if c.strip()]
    if not lang_codes:
        lang_codes = _detect_lang_codes_you_have(repo_root)
    if not lang_codes:
        # Last resort: default to the codes we know how to map from language names.
        lang_codes = sorted(set(LANG_NAME_TO_CODE.values()))

    print(f"Output dir: {out_root}")
    print(f"Keeping languages: {', '.join(lang_codes)}")
    print(f"Splits to try: {', '.join(SPLITS_TO_TRY)} (streaming={USE_STREAMING})")

    _try_hf_login_from_env()

    # Prefer loading per-language configs so we don't scan other languages.
    config_names: list[str] = []
    if get_dataset_config_names is not None:
        try:
            config_names = list(get_dataset_config_names(DATASET_ID))
        except Exception as e:
            print(f"Warning: could not list dataset config names (will fallback to global scan): {e}")
            config_names = []

    # Map config name -> lang_code, filter to only requested codes
    selected_configs: list[tuple[str, str]] = []
    for cfg in config_names:
        code = _normalize_language_to_code(cfg)
        if code and code in lang_codes:
            selected_configs.append((cfg, code))

    if selected_configs:
        cfg_msg = ", ".join([f"{cfg}->{code}" for cfg, code in selected_configs])
        print(f"Selected configs: {cfg_msg}")
    else:
        if config_names:
            print("No matching per-language configs found for requested language codes; falling back to global scan.")

    # Map (lang_code, gender, raw_speaker_id) -> state
    states: dict[tuple[str, str, str], _SpeakerShardState] = {}
    # Give stable numbering per (lang_code, gender)
    speaker_counter: dict[tuple[str, str], int] = {}

    # Stats
    total_written = 0
    total_skipped_no_lang = 0
    total_skipped_lang = 0
    total_skipped_gender = 0
    total_skipped_text = 0

    def _iter_datasets_for_split(split: str):
        if selected_configs:
            for cfg_name, _code in selected_configs:
                yield cfg_name, _code, load_dataset(DATASET_ID, name=cfg_name, split=split, streaming=USE_STREAMING)
        else:
            yield None, None, load_dataset(DATASET_ID, split=split, streaming=USE_STREAMING)

    for split in SPLITS_TO_TRY:
        try:
            datasets_for_split = list(_iter_datasets_for_split(split))
        except Exception as e:
            msg = str(e)
            if "gated dataset" in msg.lower() or "must be authenticated" in msg.lower():
                print("")
                print("ERROR: This dataset is gated. Authenticate first:")
                print("  - set HF_TOKEN env var, OR")
                print("  - run: huggingface-cli login")
                print("")
                raise
            print(f"Skipping split '{split}' (could not load): {e}")
            continue

        for cfg_name, cfg_code, ds in datasets_for_split:
            # Avoid torchcodec dependency by forcing decode=False.
            try:
                if hasattr(ds, "column_names") and "audio" in getattr(ds, "column_names", []):
                    ds = ds.cast_column("audio", Audio(decode=False))
            except Exception as e:
                print(f"Warning: could not cast audio column to decode=False (continuing): {e}")

            label = f"{split}" if not cfg_name else f"{split}:{cfg_name}"
            print(f"\nProcessing: {label}")

            total = None
            try:
                total = len(ds)  # type: ignore[arg-type]
            except Exception:
                total = None

            pbar = tqdm(ds, total=total, desc=label, unit="utt", dynamic_ncols=True)
            update_every = 200  # reduce overhead from frequent postfix updates

            for i, sample in enumerate(pbar, start=1):
                if MAX_UTTERANCES_TOTAL is not None and total_written >= MAX_UTTERANCES_TOTAL:
                    break

                # If we loaded a per-language config, trust that code.
                lang_code = cfg_code
                if not lang_code:
                    lang_val = sample.get("language") or sample.get("lang") or sample.get("Language")
                    lang_code = _normalize_language_to_code(lang_val)
                if not lang_code:
                    total_skipped_no_lang += 1
                    continue
                if lang_code not in lang_codes:
                    total_skipped_lang += 1
                    continue

                gender_val = sample.get("gender") or sample.get("Gender")
                gender = _normalize_gender(gender_val)
                if not gender:
                    total_skipped_gender += 1
                    continue

                text_val = sample.get("text") or sample.get("transcript") or sample.get("sentence")
                text = str(text_val).strip() if text_val is not None else ""
                if not text:
                    total_skipped_text += 1
                    continue

                raw_speaker = (
                    sample.get("speaker")
                    or sample.get("speaker_id")
                    or sample.get("spk_id")
                    or sample.get("speakerid")
                    or f"{lang_code}_{gender}"
                )
                raw_speaker_str = str(raw_speaker)
                key = (lang_code, gender, raw_speaker_str)

                if key not in states:
                    speaker_key = (lang_code, gender)
                    speaker_counter[speaker_key] = speaker_counter.get(speaker_key, 0) + 1
                    spk_idx = speaker_counter[speaker_key]

                    lang_name = LANG_CODE_TO_NAME.get(lang_code, lang_code)
                    speaker_dirname = f"IISc_SYSPINProject_RASA_{_safe_name(lang_name)}_{gender}_Spk{spk_idx:03d}"
                    speaker_dir = out_root / lang_code / "IISc_SYSPIN_Data" / speaker_dirname
                    (speaker_dir / "wav").mkdir(parents=True, exist_ok=True)
                    states[key] = _SpeakerShardState(
                        speaker_dir=speaker_dir,
                        lang_code=lang_code,
                        gender=gender,
                    )

                state = states[key]

                style_val = sample.get("style") or sample.get("emotion") or sample.get("domain")
                domain = str(style_val).strip().upper() if style_val is not None and str(style_val).strip() else "RASA"

                base = sample.get("filename") or sample.get("utt_id") or sample.get("id")
                base_stem = Path(str(base)).stem if base is not None else None
                audio_id = state.next_audio_id(split=split, base=base_stem)
                dst_wav = state.speaker_dir / "wav" / f"{audio_id}.wav"

                try:
                    _export_audio(sample, dst_wav)
                except Exception as e:
                    # If audio export fails, skip the sample without polluting transcripts.
                    print(f"Warning: skipping sample (audio export failed): {e}")
                    continue

                state.add(audio_id=audio_id, transcript=text, domain=domain)
                state.flush_if_needed(force=False)
                total_written += 1

                if i % update_every == 0:
                    pbar.set_postfix(
                        written=total_written,
                        speakers=len(states),
                        no_lang=total_skipped_no_lang,
                        lang_filtered=total_skipped_lang,
                        bad_gender=total_skipped_gender,
                        empty_text=total_skipped_text,
                    )

            if MAX_UTTERANCES_TOTAL is not None and total_written >= MAX_UTTERANCES_TOTAL:
                print(f"Reached MAX_UTTERANCES_TOTAL={MAX_UTTERANCES_TOTAL}, stopping early.")
                break

        if MAX_UTTERANCES_TOTAL is not None and total_written >= MAX_UTTERANCES_TOTAL:
            break

    # Final flush of all shards
    flushed = 0
    for state in states.values():
        flushed += state.flush_if_needed(force=True)

    print("\nDone.")
    print(f"Written samples: {total_written}")
    print(f"Speakers created: {len(states)}")
    print(f"Final transcript entries flushed: {flushed}")
    print(
        "Skipped: "
        f"no_lang={total_skipped_no_lang}, "
        f"lang_filtered={total_skipped_lang}, "
        f"bad_gender={total_skipped_gender}, "
        f"empty_text={total_skipped_text}"
    )
    print(f"Output root: {out_root}")

    if VERIFY_OUTPUT_AFTER_DOWNLOAD:
        _validate_syspin_layout(out_root, lang_codes)


if __name__ == "__main__":
    main()