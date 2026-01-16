"""Dataset sanity checker for IISc SYSPIN TTS data."""

import json
from pathlib import Path

import librosa

# === Config ===
DATA_DIR = "dataset"
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 30.0


def load_transcriptions(speaker_dir: Path) -> dict[str, dict]:
    """Load transcripts from *_Transcripts.json files."""
    transcripts = {}
    for json_file in speaker_dir.glob("*_Transcripts.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "Transcripts" in data:
                for key, value in data["Transcripts"].items():
                    transcripts[key] = {
                        "transcript": value.get("Transcript", ""),
                        "domain": value.get("Domain", ""),
                    }
    return transcripts


def check_wav_file(wav_path: Path) -> tuple[bool, str, float]:
    """
    Check if WAV file is valid and loadable.
    
    Returns: (is_valid, error_message, duration_sec)
    """
    try:
        audio, sr = librosa.load(wav_path, sr=None)
        duration = len(audio) / sr
        
        if duration < MIN_DURATION_SEC:
            return False, f"Too short ({duration:.2f}s < {MIN_DURATION_SEC}s)", duration
        if duration > MAX_DURATION_SEC:
            return False, f"Too long ({duration:.2f}s > {MAX_DURATION_SEC}s)", duration
        if len(audio) == 0:
            return False, "Empty audio", 0.0
            
        return True, "", duration
    except Exception as e:
        return False, f"Corrupt/unreadable: {e}", 0.0


def check_speaker_dir(speaker_dir: Path) -> dict:
    """Run sanity checks on a single speaker directory."""
    print(f"\n{'='*60}")
    print(f"Checking: {speaker_dir.name}")
    print(f"{'='*60}")
    
    results = {
        "speaker": speaker_dir.name,
        "total_wavs": 0,
        "total_transcripts": 0,
        "valid_pairs": 0,
        "missing_transcripts": [],
        "missing_wavs": [],
        "corrupt_wavs": [],
        "empty_transcripts": [],
        "duration_issues": [],
        "total_duration_sec": 0.0,
    }
    
    # Load transcripts from speaker_dir (JSON is at speaker level)
    transcripts = load_transcriptions(speaker_dir)
    results["total_transcripts"] = len(transcripts)
    
    # Get all WAV files from wav/ subdirectory
    wav_dir = speaker_dir / "wav"
    if not wav_dir.exists():
        print(f"  ⚠ No wav/ subdirectory found")
        return results
    
    wav_files = list(wav_dir.glob("*.wav"))
    results["total_wavs"] = len(wav_files)
    
    wav_ids = {w.stem for w in wav_files}
    transcript_ids = set(transcripts.keys())
    
    # Check for missing transcripts (WAVs without transcript)
    missing_transcripts = wav_ids - transcript_ids
    results["missing_transcripts"] = list(missing_transcripts)
    
    # Check for orphaned transcripts (transcripts without WAV)
    missing_wavs = transcript_ids - wav_ids
    results["missing_wavs"] = list(missing_wavs)
    
    # Check each WAV file
    for wav_path in wav_files:
        audio_id = wav_path.stem
        
        # Check if transcript exists and is non-empty
        if audio_id in transcripts:
            text = transcripts[audio_id]["transcript"]
            if not text.strip():
                results["empty_transcripts"].append(audio_id)
                continue
        else:
            continue  # Already counted in missing_transcripts
        
        # Check WAV file integrity
        is_valid, error, duration = check_wav_file(wav_path)
        
        if not is_valid:
            if "Corrupt" in error or "unreadable" in error:
                results["corrupt_wavs"].append((audio_id, error))
            else:
                results["duration_issues"].append((audio_id, error))
        else:
            results["valid_pairs"] += 1
            results["total_duration_sec"] += duration
    
    return results


def print_results(results: dict):
    """Print check results for a speaker."""
    print(f"\n  WAV files:      {results['total_wavs']}")
    print(f"  Transcripts:    {results['total_transcripts']}")
    print(f"  Valid pairs:    {results['valid_pairs']}")
    print(f"  Total duration: {results['total_duration_sec']/3600:.2f} hours")
    
    if results["missing_transcripts"]:
        print(f"\n  ⚠ Missing transcripts ({len(results['missing_transcripts'])}):")
        for x in results["missing_transcripts"][:5]:
            print(f"    - {x}")
        if len(results["missing_transcripts"]) > 5:
            print(f"    ... and {len(results['missing_transcripts'])-5} more")
    
    if results["missing_wavs"]:
        print(f"\n  ⚠ Missing WAV files ({len(results['missing_wavs'])}):")
        for x in results["missing_wavs"][:5]:
            print(f"    - {x}")
        if len(results["missing_wavs"]) > 5:
            print(f"    ... and {len(results['missing_wavs'])-5} more")
    
    if results["corrupt_wavs"]:
        print(f"\n  ❌ Corrupt WAV files ({len(results['corrupt_wavs'])}):")
        for audio_id, err in results["corrupt_wavs"][:5]:
            print(f"    - {audio_id}: {err}")
        if len(results["corrupt_wavs"]) > 5:
            print(f"    ... and {len(results['corrupt_wavs'])-5} more")
    
    if results["empty_transcripts"]:
        print(f"\n  ⚠ Empty transcripts ({len(results['empty_transcripts'])}):")
        for x in results["empty_transcripts"][:5]:
            print(f"    - {x}")
        if len(results["empty_transcripts"]) > 5:
            print(f"    ... and {len(results['empty_transcripts'])-5} more")
    
    if results["duration_issues"]:
        print(f"\n  ⚠ Duration issues ({len(results['duration_issues'])}):")
        for audio_id, err in results["duration_issues"][:5]:
            print(f"    - {audio_id}: {err}")
        if len(results["duration_issues"]) > 5:
            print(f"    ... and {len(results['duration_issues'])-5} more")


def find_speaker_dirs(data_path: Path) -> list[Path]:
    """
    Find all speaker directories.
    Structure: dataset/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_*/
    """
    speaker_dirs = []
    
    for lang_dir in data_path.iterdir():
        if not lang_dir.is_dir():
            continue
        
        syspin_data = lang_dir / "IISc_SYSPIN_Data"
        if not syspin_data.exists():
            continue
        
        for d in syspin_data.iterdir():
            if d.is_dir() and d.name.startswith("IISc_SYSPINProject_"):
                speaker_dirs.append(d)
    
    return speaker_dirs


def main():
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    
    # Find speaker directories
    speaker_dirs = find_speaker_dirs(data_path)
    
    if not speaker_dirs:
        print(f"❌ No speaker directories found in {DATA_DIR}")
        print("   Expected: dataset/<lang>/IISc_SYSPIN_Data/IISc_SYSPINProject_*/")
        return
    
    print(f"Found {len(speaker_dirs)} speaker directory(ies)")
    
    all_results = []
    for speaker_dir in sorted(speaker_dirs):
        results = check_speaker_dir(speaker_dir)
        print_results(results)
        all_results.append(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_valid = sum(r["valid_pairs"] for r in all_results)
    total_duration = sum(r["total_duration_sec"] for r in all_results)
    total_issues = sum(
        len(r["missing_transcripts"]) + 
        len(r["missing_wavs"]) + 
        len(r["corrupt_wavs"]) + 
        len(r["empty_transcripts"]) +
        len(r["duration_issues"])
        for r in all_results
    )
    
    print(f"Total valid samples: {total_valid}")
    print(f"Total duration:      {total_duration/3600:.2f} hours")
    print(f"Total issues:        {total_issues}")
    
    if total_issues == 0:
        print("\n✅ All checks passed!")
    else:
        print(f"\n⚠ Found {total_issues} issue(s) - review above for details")


if __name__ == "__main__":
    main()
