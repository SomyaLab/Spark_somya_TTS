"""Dataset loading utilities for Spark-TTS."""

import json
import hashlib
import logging
import os
import random
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable

import numpy as np
import soundfile as sf
from datasets import Dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm

from .tokenizer import AudioTokenizer

logger = logging.getLogger("spark_tts")

# Default duration limits (can be overridden via config)
DEFAULT_MIN_DURATION = 0.05   # seconds
DEFAULT_MAX_DURATION = 30.0  # seconds

def _infer_lang_from_audio_path(audio_path: str, data_dir: str) -> str | None:
    """
    Best-effort infer language code from canonical layout:
      <...>/<data_dir>/<lang>/IISc_SYSPIN_Data/...
    """
    try:
        # data_dir may be relative ("datasets") or absolute; we only need its leaf name.
        root_name = Path(str(data_dir)).name
        parts = Path(str(audio_path)).parts
        if root_name in parts:
            i = parts.index(root_name)
            if i + 1 < len(parts):
                return str(parts[i + 1])
    except Exception:
        pass
    return None


def get_audio_duration(audio_path: str) -> float | None:
    """Get audio duration in seconds without loading full audio.
    
    Returns None if file is empty/corrupted.
    """
    # Check for empty files first
    if os.path.getsize(audio_path) == 0:
        return None
    
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        # Fallback: try loading with soundfile
        try:
            audio, sr = sf.read(audio_path)
            return len(audio) / sr
        except Exception:
            return None


def load_audio_fast(audio_path: str) -> tuple:
    """Load audio using soundfile (fast and reliable for WAV).
    
    Returns:
        Tuple of (audio_array as numpy, sample_rate)
    """
    audio, sr = sf.read(audio_path)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def preprocess_samples_batched(
    samples: list[dict],
    audio_tokenizer: AudioTokenizer,
    batch_size: int = 16,
    num_workers: int = 4,
) -> list[dict]:
    """
    Process samples in batches for faster tokenization.
    
    Uses parallel audio loading and batched GPU tokenization.
    
    Args:
        samples: List of sample dicts with audio_path, text
        audio_tokenizer: AudioTokenizer instance
        batch_size: Number of samples to process in each GPU batch
        num_workers: Number of parallel workers for audio loading
        
    Returns:
        List of processed dicts with prompt and completion
    """
    processed = []
    errors = 0
    logged_errors = 0
    
    def load_single(sample):
        """Load a single audio file (for parallel execution)."""
        try:
            audio, sr = load_audio_fast(sample["audio_path"])
            if len(audio) == 0:
                return None
            return {
                "audio": audio,
                "sr": sr,
                "text": sample["text"],
                "audio_path": sample["audio_path"],
            }
        except Exception:
            return None
    
    pbar = tqdm(total=len(samples), desc="Tokenizing audio (batched)", unit="file")
    
    # Reuse the thread pool across batches (creating it per-batch is expensive).
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in range(0, len(samples), batch_size):
            batch_samples = samples[batch_start:batch_start + batch_size]
            
            # Parallel audio loading
            loaded = list(executor.map(load_single, batch_samples))

            # Filter out failed loads
            valid_loaded = [l for l in loaded if l is not None]
            batch_errors = len(loaded) - len(valid_loaded)
            errors += batch_errors

            if not valid_loaded:
                pbar.update(len(batch_samples))
                continue

            # Batched tokenization
            try:
                audio_arrays = [l["audio"] for l in valid_loaded]
                sampling_rates = [l["sr"] for l in valid_loaded]

                token_results = audio_tokenizer.tokenize_audio_batch(audio_arrays, sampling_rates)

                # Build output dicts
                for item, (global_tokens, semantic_tokens) in zip(valid_loaded, token_results):
                    if not global_tokens or not semantic_tokens:
                        errors += 1
                        continue

                    prompt = "".join([
                        "<|task_tts|>",
                        "<|start_content|>",
                        item["text"],
                        "<|end_content|>",
                    ])

                    completion = "".join([
                        "<|start_global_token|>",
                        global_tokens,
                        "<|end_global_token|>",
                        "<|start_semantic_token|>",
                        semantic_tokens,
                        "<|end_semantic_token|>",
                        "<|im_end|>",
                    ])

                    processed.append({"prompt": prompt, "completion": completion})

            except Exception as e:
                # Fallback: skip this batch
                errors += len(valid_loaded)
                if logged_errors < 3:
                    logger.warning(f"Batch tokenization failed, skipping batch: {e}")
                    logged_errors += 1

            pbar.update(len(batch_samples))
    
    pbar.close()
    return processed, errors


def preprocess_samples_batched_ids(
    samples: list[dict],
    audio_tokenizer: AudioTokenizer,
    batch_size: int = 16,
    num_workers: int = 4,
    on_batch: Callable[[list[dict]], None] | None = None,
    collect: bool = True,
) -> tuple[list[dict], int]:
    """
    Process samples in batches and return raw audio token IDs.

    Output records contain:
      - audio_path, text, speaker, duration
      - global_ids: list[int]
      - semantic_ids: list[int]
    """
    processed: list[dict] = []
    errors = 0
    logged_errors = 0

    def load_single(sample: dict) -> dict | None:
        try:
            audio, sr = load_audio_fast(sample["audio_path"])
            if len(audio) == 0:
                return None
            return {
                "audio": audio,
                "sr": sr,
                "audio_path": sample.get("audio_path"),
                "text": sample.get("text"),
                "speaker": sample.get("speaker"),
                "duration": sample.get("duration"),
            }
        except Exception:
            return None

    pbar = tqdm(total=len(samples), desc="Tokenizing audio IDs (batched)", unit="file")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in range(0, len(samples), batch_size):
            batch_samples = samples[batch_start : batch_start + batch_size]
            loaded = list(executor.map(load_single, batch_samples))

            valid_loaded = [l for l in loaded if l is not None]
            batch_errors = len(loaded) - len(valid_loaded)
            errors += batch_errors

            if not valid_loaded:
                pbar.update(len(batch_samples))
                continue

            def _emit_rows(items: list[dict], token_results: list[tuple[list[int], list[int]]]) -> None:
                batch_rows: list[dict] = []
                for item, (global_ids, semantic_ids) in zip(items, token_results):
                    if not global_ids or not semantic_ids:
                        continue
                    row = {
                        "audio_path": item["audio_path"],
                        "text": item["text"],
                        "speaker": item["speaker"],
                        "duration": item["duration"],
                        "global_ids": [int(x) for x in global_ids],
                        "semantic_ids": [int(x) for x in semantic_ids],
                    }
                    if collect:
                        processed.append(row)
                    batch_rows.append(row)

                if on_batch is not None and batch_rows:
                    try:
                        on_batch(batch_rows)
                    except Exception as e:
                        if logged_errors < 3:
                            logger.warning(f"on_batch callback failed (ignored): {e}")
                            logged_errors += 1

            # Tokenize with OOM-aware retries: shrink microbatch instead of skipping.
            items = valid_loaded
            micro = len(items)
            while items:
                micro = min(micro, len(items))
                try:
                    token_results = audio_tokenizer.tokenize_audio_batch_ids(
                        [l["audio"] for l in items[:micro]],
                        [l["sr"] for l in items[:micro]],
                    )
                    _emit_rows(items[:micro], token_results)
                    items = items[micro:]
                    micro = min(micro, len(items)) if items else micro
                except Exception as e:
                    msg = str(e).lower()
                    is_oom = ("out of memory" in msg) or ("cuda oom" in msg)
                    if is_oom:
                        try:
                            import torch, gc  # local import to avoid overhead when unused

                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if micro > 1:
                            micro = max(1, micro // 2)
                            if logged_errors < 3:
                                logger.warning(f"CUDA OOM in tokenization; retrying with microbatch={micro}")
                                logged_errors += 1
                            continue
                    # Non-OOM or microbatch=1 still failing: count as error and skip this one item.
                    errors += micro
                    if logged_errors < 3:
                        logger.warning(f"Tokenization failed for microbatch={micro}, skipping: {e}")
                        logged_errors += 1
                    items = items[micro:]

            pbar.update(len(batch_samples))

    pbar.close()
    return processed, errors


def _require_token_id(tokenizer, token: str, cache: dict[str, int]) -> int:
    token_id = cache.get(token)
    if token_id is None:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        cache[token] = token_id
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not know token: {token}")
    return token_id


def build_cloning_pairs_from_tokens(
    utterances: list[dict],
    pairs_per_speaker: int = 5,
) -> list[dict]:
    """Build cross-utterance cloning pairs using cached token IDs (no re-tokenization)."""
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for u in utterances:
        spk = u.get("speaker")
        if spk:
            by_speaker[str(spk)].append(u)

    multi = {s: utts for s, utts in by_speaker.items() if len(utts) >= 2}
    if not multi:
        logger.warning("No speakers with multiple utterances for cloning pairs")
        return []

    logger.info(f"Building cloning pairs from cached tokens for {len(multi)} speakers")

    out: list[dict] = []
    for speaker, utts in tqdm(multi.items(), desc="Building cloning pairs", unit="speaker"):
        num_pairs = min(pairs_per_speaker, len(utts) * (len(utts) - 1) // 2)
        for _ in range(num_pairs):
            ref, target = random.sample(utts, 2)
            out.append(
                {
                    "text": target["text"],
                    "speaker": speaker,
                    "duration": target.get("duration"),
                    "global_ids": ref["global_ids"],
                    "semantic_ids": target["semantic_ids"],
                    "is_cloning_pair": True,
                }
            )

    logger.info(f"Generated {len(out)} cloning pairs")
    return out


def tokenize_for_training_ids(
    samples: list[dict],
    tokenizer,
    max_seq_length: int = 2048,
    *,
    train_objective: str = "clone_semantic",
) -> list[dict]:
    """
    Build training rows directly from token IDs (no prompt/completion string re-tokenization for audio tokens).

    Expected input sample fields:
      - text: str
      - global_ids: list[int]  (BiCodec code ids)
      - semantic_ids: list[int] (BiCodec code ids)
    """
    token_cache: dict[str, int] = {}

    tok_task = "<|task_tts|>"
    tok_start_content = "<|start_content|>"
    tok_end_content = "<|end_content|>"
    tok_start_global = "<|start_global_token|>"
    tok_end_global = "<|end_global_token|>"
    tok_start_sem = "<|start_semantic_token|>"
    tok_end_sem = "<|end_semantic_token|>"
    tok_im_end = "<|im_end|>"

    # Validate required marker tokens exist.
    _ = _require_token_id(tokenizer, tok_task, token_cache)
    _ = _require_token_id(tokenizer, tok_start_content, token_cache)
    _ = _require_token_id(tokenizer, tok_end_content, token_cache)
    start_global_id = _require_token_id(tokenizer, tok_start_global, token_cache)
    end_global_id = _require_token_id(tokenizer, tok_end_global, token_cache)
    start_sem_id = _require_token_id(tokenizer, tok_start_sem, token_cache)
    end_sem_id = _require_token_id(tokenizer, tok_end_sem, token_cache)
    im_end_id = _require_token_id(tokenizer, tok_im_end, token_cache)

    tokenized_samples: list[dict] = []
    skipped_count = 0
    logged_errors = 0

    for sample in tqdm(samples, desc="Building input_ids/labels", unit="sample"):
        try:
            text = sample.get("text", "")
            if not str(text).strip():
                raise ValueError("Empty text")

            global_ids = sample.get("global_ids") or []
            semantic_ids = sample.get("semantic_ids") or []
            if not global_ids or not semantic_ids:
                raise ValueError("Missing audio token IDs")

            # Prompt is short/cheap; keep as string encode for correctness.
            prompt = "".join([tok_task, tok_start_content, str(text), tok_end_content])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            global_token_ids = [
                _require_token_id(tokenizer, f"<|bicodec_global_{i}|>", token_cache)
                for i in global_ids
            ]
            semantic_token_ids = [
                _require_token_id(tokenizer, f"<|bicodec_semantic_{i}|>", token_cache)
                for i in semantic_ids
            ]

            if str(train_objective).lower() in {"clone_semantic", "clone_semantic_v1"}:
                # Clone-conditioned objective:
                # - globals are provided as context (speaker identity)
                # - loss is computed on semantic tokens only (plus end markers)
                context_ids = (
                    prompt_ids
                    + [start_global_id]
                    + global_token_ids
                    + [end_global_id]
                    + [start_sem_id]
                )
                completion_ids = semantic_token_ids + [end_sem_id] + [im_end_id]

                if len(context_ids) >= max_seq_length:
                    raise ValueError("Context exceeds max_seq_length (no room for semantic targets)")

                # Truncate semantic completion if needed.
                max_completion = max_seq_length - len(context_ids)
                if max_completion <= 2:
                    raise ValueError("No room for end markers under max_seq_length")
                # Keep end markers always: truncate only semantic ids, then append [end_sem, im_end].
                sem_keep = max_completion - 2
                sem_ids_trunc = semantic_token_ids[:sem_keep]
                if not sem_ids_trunc:
                    raise ValueError("No semantic targets after truncation")
                completion_ids = sem_ids_trunc + [end_sem_id] + [im_end_id]

                input_ids = context_ids + completion_ids
                labels = [-100] * len(context_ids) + completion_ids.copy()
                attention_mask = [1] * len(input_ids)
            else:
                # Back-compat: original objective (model predicts globals + semantics).
                completion_ids = (
                    [start_global_id]
                    + global_token_ids
                    + [end_global_id]
                    + [start_sem_id]
                    + semantic_token_ids
                    + [end_sem_id]
                    + [im_end_id]
                )

                input_ids = prompt_ids + completion_ids
                prompt_len = len(prompt_ids)

                if len(input_ids) > max_seq_length:
                    if prompt_len >= max_seq_length:
                        prompt_len = max_seq_length
                        input_ids = input_ids[:max_seq_length]
                    else:
                        input_ids = input_ids[:max_seq_length]

                if prompt_len > len(input_ids):
                    prompt_len = len(input_ids)

                labels = [-100] * prompt_len + input_ids[prompt_len:].copy()
                assert len(labels) == len(input_ids)
                attention_mask = [1] * len(input_ids)

            tokenized_samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        except Exception as e:
            skipped_count += 1
            if logged_errors < 3:
                logger.warning(f"Error building training row: {e}")
                logged_errors += 1
            continue

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} samples during training-row build")

    return tokenized_samples


def preprocess_sample(
    audio_path: str,
    text: str,
    audio_tokenizer: AudioTokenizer,
) -> dict[str, str]:
    """
    Preprocess a single audio sample for training.
    Returns dict with 'prompt' and 'completion' for loss masking.
    
    Note: Speaker identity is encoded in global tokens, not text prefix.
    This ensures train/inference consistency for zero-shot cloning.
    
    Args:
        audio_path: Path to audio file
        text: Transcript text
        audio_tokenizer: AudioTokenizer instance
        
    Returns:
        Dict with 'prompt' and 'completion' keys
        
    Raises:
        ValueError: If audio loading or tokenization fails
    """
    # Back-compat helper: not used by the current pipeline.
    if not text or not text.strip():
        raise ValueError(f"Empty text for audio: {audio_path}")
    
    try:
        audio_array, sr = load_audio_fast(audio_path)
        if len(audio_array) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to load audio {audio_path}: {e}") from e
    
    try:
        global_tokens, semantic_tokens = audio_tokenizer.tokenize_audio(audio_array, sr)
        if not global_tokens or not semantic_tokens:
            raise ValueError(f"Empty tokens from audio: {audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to tokenize audio {audio_path}: {e}") from e

    # Prompt: text content (loss will be masked)
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
    ])
    
    # Completion: audio tokens (loss will be computed)
    completion = "".join([
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|end_semantic_token|>",
        "<|im_end|>",
    ])

    return {"prompt": prompt, "completion": completion}


def preprocess_cloning_pair(
    ref_audio_path: str,
    target_audio_path: str,
    target_text: str,
    audio_tokenizer: AudioTokenizer,
) -> dict[str, str]:
    """
    Preprocess a cross-utterance cloning pair for zero-shot training.
    
    Uses global tokens from ref_audio (speaker A) and text + semantic tokens 
    from target_audio (same speaker, different utterance B).
    
    This teaches the model to use global tokens as speaker identity 
    independent of the actual utterance content.
    
    Args:
        ref_audio_path: Path to reference audio (for speaker identity)
        target_audio_path: Path to target audio (for content)
        target_text: Text transcript for target audio
        audio_tokenizer: AudioTokenizer instance
        
    Returns:
        Dict with 'prompt' and 'completion' keys
        
    Raises:
        ValueError: If audio loading or tokenization fails
    """
    # Back-compat helper: not used by the current pipeline.
    if not target_text or not target_text.strip():
        raise ValueError(f"Empty text for target audio: {target_audio_path}")
    
    # Extract global tokens from reference audio (speaker identity)
    try:
        ref_audio, ref_sr = load_audio_fast(ref_audio_path)
        if len(ref_audio) == 0:
            raise ValueError(f"Empty reference audio: {ref_audio_path}")
        ref_global_tokens, _ = audio_tokenizer.tokenize_audio(ref_audio, ref_sr)
        if not ref_global_tokens:
            raise ValueError(f"Empty global tokens from reference: {ref_audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to process reference audio {ref_audio_path}: {e}") from e
    
    # Extract semantic tokens from target audio (content)
    try:
        target_audio, target_sr = load_audio_fast(target_audio_path)
        if len(target_audio) == 0:
            raise ValueError(f"Empty target audio: {target_audio_path}")
        _, target_semantic_tokens = audio_tokenizer.tokenize_audio(target_audio, target_sr)
        if not target_semantic_tokens:
            raise ValueError(f"Empty semantic tokens from target: {target_audio_path}")
    except Exception as e:
        raise ValueError(f"Failed to process target audio {target_audio_path}: {e}") from e
    
    # Prompt: text content (loss will be masked)
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        target_text,
        "<|end_content|>",
    ])
    
    # Completion: ref's global + target's semantic (loss will be computed)
    completion = "".join([
        "<|start_global_token|>",
        ref_global_tokens,  # From reference (speaker identity)
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        target_semantic_tokens,  # From target (content)
        "<|end_semantic_token|>",
        "<|im_end|>",
    ])
    
    return {"prompt": prompt, "completion": completion}


def build_cloning_pairs(
    samples: list[dict],
    audio_tokenizer: AudioTokenizer,
    max_seq_length: int = 2048,
    pairs_per_speaker: int = 5,
) -> list[dict]:
    """
    Build cross-utterance cloning pairs from samples grouped by speaker.
    
    For each speaker with multiple utterances, creates pairs where:
    - Global tokens come from one utterance (reference)
    - Text + semantic tokens come from another utterance (target)
    
    Args:
        samples: List of sample dicts with audio_path, text, speaker
        audio_tokenizer: AudioTokenizer instance
        max_seq_length: Maximum sequence length for filtering
        pairs_per_speaker: Number of cloning pairs to generate per speaker
        
    Returns:
        List of processed cloning pair dicts
    """
    # Back-compat helper: not used by the current pipeline.
    # Group samples by speaker
    by_speaker = defaultdict(list)
    for sample in samples:
        by_speaker[sample["speaker"]].append(sample)
    
    # Filter to speakers with at least 2 utterances
    multi_utterance_speakers = {
        speaker: utts for speaker, utts in by_speaker.items() 
        if len(utts) >= 2
    }
    
    if not multi_utterance_speakers:
        logger.warning("No speakers with multiple utterances for cloning pairs")
        return []
    
    logger.info(f"Building cloning pairs from {len(multi_utterance_speakers)} speakers")
    
    cloning_samples = []
    for speaker, utterances in tqdm(multi_utterance_speakers.items(), desc="Building cloning pairs"):
        # Generate random pairs for this speaker
        num_pairs = min(pairs_per_speaker, len(utterances) * (len(utterances) - 1) // 2)
        
        for _ in range(num_pairs):
            # Pick two different utterances
            ref, target = random.sample(utterances, 2)
            
            try:
                tokenized = preprocess_cloning_pair(
                    ref_audio_path=ref["audio_path"],
                    target_audio_path=target["audio_path"],
                    target_text=target["text"],
                    audio_tokenizer=audio_tokenizer,
                )
                cloning_samples.append(tokenized)
            except Exception as e:
                logger.warning(f"Skipping cloning pair {ref['audio_path']} -> {target['audio_path']}: {e}")
    
    logger.info(f"Generated {len(cloning_samples)} cloning pairs")
    return cloning_samples


def tokenize_for_training(
    samples: list[dict],
    tokenizer,
    max_seq_length: int = 2048,
) -> list[dict]:
    """
    Tokenize samples and create labels with -100 for prompt tokens.
    
    This enables completion-only loss: loss is only computed on audio tokens,
    not on the text prompt.
    
    Args:
        samples: List of dicts with 'prompt' and 'completion' keys
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length (will truncate if exceeded)
        
    Returns:
        List of dicts with 'input_ids', 'attention_mask', 'labels'
    """
    # Back-compat helper: not used by the current pipeline.
    tokenized_samples = []
    skipped_count = 0
    logged_errors = 0
    
    for sample in tqdm(samples, desc="Tokenizing for training", unit="sample"):
        try:
            prompt = sample["prompt"]
            completion = sample["completion"]
            
            # Tokenize prompt and completion separately
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            
            # Combine
            input_ids = prompt_ids + completion_ids
            prompt_len = len(prompt_ids)
            
            # Truncate if needed (preserve as much prompt as possible, truncate from completion)
            if len(input_ids) > max_seq_length:
                # If prompt is longer than max_seq_length, truncate prompt too
                if prompt_len >= max_seq_length:
                    # Entire sequence is prompt (edge case - should be filtered earlier)
                    prompt_len = max_seq_length
                    input_ids = input_ids[:max_seq_length]
                else:
                    # Truncate from completion side, keep full prompt
                    input_ids = input_ids[:max_seq_length]
                    # prompt_len stays the same
            
            # Ensure labels match input_ids length exactly
            # Create labels: -100 for prompt tokens (no loss), actual IDs for completion tokens
            labels = [-100] * prompt_len + input_ids[prompt_len:].copy()
            
            # Verify lengths match
            assert len(labels) == len(input_ids), \
                f"Label length {len(labels)} != input_ids length {len(input_ids)}"
            
            # Attention mask (all 1s for now, padding handled by collator)
            attention_mask = [1] * len(input_ids)
            
            tokenized_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        except Exception as e:
            skipped_count += 1
            # Avoid spamming logs on large datasets
            if logged_errors < 3:
                logger.warning(f"Error tokenizing sample: {e}")
                logged_errors += 1
            continue
    
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} samples during tokenization")
    
    return tokenized_samples


def get_wav_files(speaker_dir: Path) -> list[Path]:
    """Get all WAV files from speaker's wav/ subdirectory."""
    wav_dir = speaker_dir / "wav"
    if wav_dir.exists():
        # Sorting 100k+ paths can be expensive; caller can sort later if needed.
        return list(wav_dir.glob("*.wav"))
    return []


def load_transcriptions(data_dir: Path) -> dict[str, dict]:
    """Load transcriptions from the IISc SYSPIN format."""
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
    tokenizer,
    speaker_name: str | None = None,
    cache_dir: str = ".cache/tokenized_dataset",
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    max_seq_length: int = 2048,
    model_dir: str | None = None,
    sample_rate: int = 16000,
    use_cloning_pairs: bool = True,
    cloning_pairs_per_speaker: int = 5,
    sample_limit: int | None = None,
    languages: list[str] | None = None,
    language_sampling: str = "proportional",
    base_languages: list[str] | None = None,
    max_base_fraction: float = 0.30,
    max_samples_per_language: int | None = None,
    seed: int = 42,
    num_loading_workers: int = 4,
    tokenizer_batch_size: int = 8,
    train_objective: str = "clone_semantic",
    clone_cross_prob: float = 0.8,
) -> Dataset:
    """
    Load local dataset from IISc SYSPIN data structure.
    Returns pre-tokenized HuggingFace Dataset with input_ids, attention_mask, labels.
    Labels have -100 for prompt tokens (loss masked) and actual IDs for completion.
    
    Args:
        data_dir: Path to dataset directory
        audio_tokenizer: AudioTokenizer instance for audio-to-tokens
        tokenizer: HuggingFace tokenizer for text tokenization
        speaker_name: Optional filter for specific speaker
        cache_dir: Directory for caching tokenized dataset
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        max_seq_length: Maximum sequence length (for post-tokenization filter)
        model_dir: Model directory (for cache key)
        sample_rate: Audio sample rate (for cache key)
        use_cloning_pairs: Whether to use cross-utterance globals (speaker identity)
        cloning_pairs_per_speaker: Number of cloning pairs per speaker
        sample_limit: Maximum number of samples to process (for testing)
        languages: Optional list of dataset language codes (folder names under data_dir) to include.
        language_sampling: "proportional" or "balanced" (caps base-language share).
        base_languages: Language codes treated as base pretrained languages (eg. ["en","zh"]) for balancing.
        max_base_fraction: When language_sampling="balanced", cap base-language share to this fraction.
        max_samples_per_language: Optional hard cap per language (after filtering), to bound training cost.
        seed: Random seed for deterministic shuffles/selections.
        num_loading_workers: Number of parallel workers for audio loading
        tokenizer_batch_size: Batch size for GPU tokenization
        train_objective: Training objective. Default: \"clone_semantic\" (semantic-only loss).
        clone_cross_prob: Probability of using cross-utterance global tokens (same speaker) instead of self globals.
    """
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)

    # Fast path: if "balanced" sampling is requested but the base languages are not present
    # under the dataset root, avoid expensive lang inference.
    try:
        sampling_norm = str(language_sampling).lower().strip()
    except Exception:
        sampling_norm = "proportional"
    try:
        base_set0 = set(str(x) for x in (base_languages or []))
    except Exception:
        base_set0 = set()
    if sampling_norm == "balanced" and base_set0:
        try:
            available_langs = set()
            if data_path.exists():
                for d in data_path.iterdir():
                    if not d.is_dir():
                        continue
                    if (d / "IISc_SYSPIN_Data").exists():
                        available_langs.add(d.name)
            if available_langs and not (base_set0 & available_langs):
                language_sampling = "proportional"
        except Exception:
            # If anything goes wrong, fall back to the user's requested sampling mode.
            pass
    
    # Include all params in cache key for proper invalidation
    # Bump when the (input_ids, labels) construction changes.
    objective_v = 2
    langs_norm = ",".join(sorted(set(str(x) for x in (languages or [])))) if languages else "ALL"
    base_norm = ",".join(sorted(set(str(x) for x in (base_languages or [])))) if base_languages else ""
    cache_str = (
        f"objv{objective_v}:{data_dir}:{speaker_name}:{model_dir}:{min_duration}:{max_duration}:"
        f"{max_seq_length}:{sample_rate}:{train_objective}:{clone_cross_prob}:"
        f"{use_cloning_pairs}:{cloning_pairs_per_speaker}:{sample_limit}:"
        f"langs={langs_norm}:ls={str(language_sampling).lower()}:base={base_norm}:"
        f"maxbase={float(max_base_fraction)}:cap={max_samples_per_language}:seed={seed}"
    )
    cache_key = hashlib.md5(cache_str.encode()).hexdigest()[:12]
    dataset_cache = cache_path / cache_key
    
    if dataset_cache.exists():
        logger.info(f"Loading cached dataset from {dataset_cache}")
        cached_dataset = load_from_disk(str(dataset_cache))
        logger.info(f"Cached dataset size: {len(cached_dataset)} samples")
        if len(cached_dataset) < 100:
            logger.warning(f"Cached dataset seems unusually small ({len(cached_dataset)} samples)")
            logger.warning("Consider clearing cache if this is unexpected")
        return cached_dataset

    # Manifest cache: lists of valid (audio_path, text, speaker, duration) after filtering.
    # This isolates the slow full-corpus scan from the expensive GPU tokenization.
    manifest_root = cache_path.parent / "manifests"
    manifest_version = 1
    manifest_str = f"v{manifest_version}:{data_dir}:{speaker_name}:{min_duration}:{max_duration}"
    manifest_key = hashlib.md5(manifest_str.encode()).hexdigest()[:12]
    manifest_cache = manifest_root / manifest_key

    # Utterance-level cache: expensive audio->tokenization results (independent of max_seq_length, cloning, masking).
    utter_cache_root = cache_path.parent / "utterance_tokens"
    # IMPORTANT: keep this key stable across runs so cached IDs are actually reused.
    # Do NOT include sample_limit or the whole tokenizer config (can contain volatile fields).
    tok_fp = {
        "model_dir": str(model_dir),
        "sample_rate": int(sample_rate),
        "tokenizer_sr": int(getattr(audio_tokenizer, "sample_rate", sample_rate)),
        "volume_normalize": bool(getattr(audio_tokenizer, "config", {}).get("volume_normalize", False)),
    }
    utter_cache_version = 2
    utter_cache_str = f"v{utter_cache_version}:{manifest_key}:{json.dumps(tok_fp, sort_keys=True)}"
    utter_cache_key = hashlib.md5(utter_cache_str.encode()).hexdigest()[:12]
    utter_cache = utter_cache_root / utter_cache_key
    
    # Load or build the sample manifest (can be large; cached to avoid re-scanning corpus).
    # IMPORTANT: keep it as a HF Dataset (do NOT materialize into Python lists for full corpus).
    manifest_ds: Dataset
    if manifest_cache.exists():
        logger.info(f"Loading cached manifest from {manifest_cache}")
        manifest_ds = load_from_disk(str(manifest_cache))
        logger.info(f"Manifest samples: {len(manifest_ds)}")
    else:
        samples: list[dict] = []
        duration_filtered = 0
        no_transcript = 0
        empty_text = 0

        # If we're only sampling for a quick run, avoid scanning the entire corpus.
        # NOTE: This changes *which* samples are selected under sample_limit, but keeps the
        # guarantee that we return <= sample_limit utterances (before cloning augmentation).
        stop_early = bool(sample_limit is not None and sample_limit > 0)

        # Find all speaker directories
        # IMPORTANT:
        # - For full runs (stop_early=False), we always scan/cache the full corpus to a shared manifest key.
        # - For stop_early/sample_limit runs, we may restrict scanning to requested languages for speed,
        #   since we do NOT persist that partial manifest to the shared cache.
        languages_set = set(str(x) for x in (languages or [])) if languages else None
        speaker_dirs: list[Path] = []
        for lang_dir in data_path.iterdir():
            if not lang_dir.is_dir():
                continue
            if stop_early and languages_set is not None and lang_dir.name not in languages_set:
                continue
            syspin_data = lang_dir / "IISc_SYSPIN_Data"
            if not syspin_data.exists():
                continue
            for d in syspin_data.iterdir():
                if d.is_dir() and d.name.startswith("IISc_SYSPINProject_"):
                    if speaker_name is None or speaker_name in d.name:
                        speaker_dirs.append(d)

        # For stop_early runs, keep selection deterministic so small caches are reusable.
        if stop_early:
            try:
                speaker_dirs.sort(key=lambda p: str(p))
            except Exception:
                pass

        logger.info(f"Scanning dataset for valid samples (parallel_workers={num_loading_workers})...")
        scan_start = time.time()

        # Pre-list wav files per speaker once so we can show a global progress bar.
        wavs_by_speaker: list[tuple[Path, list[Path]]] = []
        total_wavs = 0
        for speaker_dir in speaker_dirs:
            wav_files = get_wav_files(speaker_dir)
            if stop_early:
                try:
                    wav_files.sort(key=lambda p: str(p))
                except Exception:
                    pass
            wavs_by_speaker.append((speaker_dir, wav_files))
            total_wavs += len(wav_files)

        pbar = tqdm(total=total_wavs, desc="Scanning corpus", unit="file")
        try:
            # If stop_early is set, do a simple sequential scan so we can actually stop early
            # without the thread pool needing to finish all scheduled work.
            if stop_early:
                for speaker_dir, wav_files in wavs_by_speaker:
                    transcripts = load_transcriptions(speaker_dir)
                    for wav_path in wav_files:
                        pbar.update(1)
                        audio_id = wav_path.stem
                        t = transcripts.get(audio_id)
                        if t is None:
                            no_transcript += 1
                            continue
                        text = str(t.get("transcript", ""))
                        if not text.strip():
                            empty_text += 1
                            continue
                        duration = get_audio_duration(str(wav_path))
                        if duration is None or duration < min_duration or duration > max_duration:
                            duration_filtered += 1
                            continue
                        lang = None
                        try:
                            lang = speaker_dir.parent.parent.name
                        except Exception:
                            lang = None
                        samples.append(
                            {
                                "audio_path": str(wav_path),
                                "text": text,
                                "speaker": speaker_dir.name,
                                "duration": float(duration),
                                "lang": lang,
                            }
                        )
                        if len(samples) >= int(sample_limit):
                            break
                    if len(samples) >= int(sample_limit):
                        break
            else:
                with ThreadPoolExecutor(max_workers=max(int(num_loading_workers), 1)) as executor:
                    for speaker_dir, wav_files in wavs_by_speaker:
                        transcripts = load_transcriptions(speaker_dir)

                        def process_wav(wav_path: Path) -> tuple[str, dict | None]:
                            audio_id = wav_path.stem
                            t = transcripts.get(audio_id)
                            if t is None:
                                return "no_transcript", None
                            text = str(t.get("transcript", ""))
                            if not text.strip():
                                return "empty_text", None
                            duration = get_audio_duration(str(wav_path))
                            if duration is None:
                                return "duration_filtered", None
                            if duration < min_duration or duration > max_duration:
                                return "duration_filtered", None
                            return (
                                "ok",
                                {
                                    "audio_path": str(wav_path),
                                    "text": text,
                                    "speaker": speaker_dir.name,
                                    "duration": float(duration),
                                    "lang": (speaker_dir.parent.parent.name if speaker_dir is not None else None),
                                },
                            )

                        for status, sample in executor.map(process_wav, wav_files):
                            pbar.update(1)
                            if status == "ok" and sample is not None:
                                samples.append(sample)
                            elif status == "no_transcript":
                                no_transcript += 1
                            elif status == "empty_text":
                                empty_text += 1
                            else:
                                duration_filtered += 1
        finally:
            pbar.close()

        elapsed_scan = time.time() - scan_start
        logger.info(
            "Found %d samples (filtered: no_transcript=%d, empty_text=%d, duration=%d) in %.1fs",
            len(samples),
            no_transcript,
            empty_text,
            duration_filtered,
            elapsed_scan,
        )

        if not samples:
            raise ValueError(f"No valid samples found in {data_dir}")

        # Cache manifest for fast future runs (full corpus).
        # IMPORTANT: If `sample_limit` is set, we may stop scanning early. Do NOT write that
        # partial manifest to the shared manifest cache key (which does not include sample_limit),
        # otherwise future full runs will incorrectly load a tiny manifest.
        try:
            manifest_root.mkdir(parents=True, exist_ok=True)
            manifest_ds = Dataset.from_list(samples)
            if not stop_early:
                manifest_ds.save_to_disk(str(manifest_cache))
                logger.info(f"Cached manifest to {manifest_cache}")
            else:
                logger.info("Not caching manifest (stop_early/sample_limit run)")
        except Exception as e:
            logger.warning(f"Failed to cache manifest: {e}")
            manifest_ds = Dataset.from_list(samples)

    # =========================
    # Language filtering / sampling (selection-only; does NOT affect utterance-token cache key)
    # =========================
    need_lang = (languages is not None) or (max_samples_per_language is not None) or (str(language_sampling).lower() != "proportional")
    if need_lang:
        if "lang" not in (manifest_ds.column_names or []):
            logger.info("Manifest has no 'lang' column; inferring language from audio_path...")

            def _add_lang(ex: dict) -> dict:
                ap = ex.get("audio_path", "")
                return {"lang": _infer_lang_from_audio_path(str(ap), data_dir)}

            manifest_ds = manifest_ds.map(_add_lang, desc="Inferring lang", num_proc=1)

        # Filter to requested languages
        if languages is not None:
            langs_set = set(str(x) for x in languages)
            before = len(manifest_ds)
            manifest_ds = manifest_ds.filter(lambda ex: ex.get("lang") in langs_set, desc="Filter languages")
            logger.info("Language filter applied: %s -> %s samples (langs=%s)", before, len(manifest_ds), sorted(langs_set))

        # Optional selection (shuffle + caps)
        sampling = str(language_sampling).lower().strip()
        do_shuffle = (max_samples_per_language is not None) or (sampling == "balanced")
        if do_shuffle and len(manifest_ds) > 1:
            manifest_ds = manifest_ds.shuffle(seed=int(seed))

        base_set = set(str(x) for x in (base_languages or []))
        base_max = None
        if sampling == "balanced" and base_set and 0.0 < float(max_base_fraction) < 1.0 and len(manifest_ds) > 0:
            base_cnt = 0
            other_cnt = 0
            for ex in manifest_ds:
                if ex.get("lang") in base_set:
                    base_cnt += 1
                else:
                    other_cnt += 1
            if other_cnt > 0:
                base_max = int((float(max_base_fraction) / (1.0 - float(max_base_fraction))) * other_cnt)
                logger.info(
                    "Balanced sampling: base=%s other=%s -> cap base to <=%s (max_base_fraction=%.2f)",
                    base_cnt,
                    other_cnt,
                    base_max,
                    float(max_base_fraction),
                )
            else:
                logger.info("Balanced sampling requested but no non-base languages found; keeping base languages unchanged.")

        if base_max is not None or max_samples_per_language is not None:
            keep: list[int] = []
            base_kept = 0
            per_lang_kept: dict[str, int] = defaultdict(int)
            per_cap = int(max_samples_per_language) if (max_samples_per_language is not None) else None
            for i, ex in enumerate(manifest_ds):
                lang = ex.get("lang")
                lang = str(lang) if lang is not None else ""
                if per_cap is not None:
                    if per_lang_kept[lang] >= per_cap:
                        continue
                if base_max is not None and lang in base_set:
                    if base_kept >= int(base_max):
                        continue
                    base_kept += 1
                per_lang_kept[lang] += 1
                keep.append(i)
            before = len(manifest_ds)
            manifest_ds = manifest_ds.select(keep)
            logger.info("Applied language caps: %s -> %s samples", before, len(manifest_ds))

    # Apply sample limit for testing (after language selection)
    if sample_limit is not None and sample_limit > 0:
        original_count = len(manifest_ds)
        manifest_ds = manifest_ds.select(range(min(int(sample_limit), original_count)))
        logger.info(f"Applied sample_limit: {original_count} → {len(manifest_ds)} samples")

    # If we only need a small subset, avoid loading huge shard caches into memory.
    subset_needed_paths: set[str] | None = None
    try:
        if sample_limit is not None and int(sample_limit) > 0 and len(manifest_ds) <= 5000:
            subset_needed_paths = set(str(x) for x in manifest_ds["audio_path"])
    except Exception:
        subset_needed_paths = None

    # Sort by duration before batching to reduce padding waste in GPU batches.
    # This can significantly improve throughput for wav2vec2 + BiCodec tokenization.
    try:
        manifest_ds = manifest_ds.sort("duration")
    except Exception:
        pass

    # Load or build utterance token cache (expensive step).
    # Cache is keyed by audio_path so subsets (`--limit`) reuse previously tokenized utterances.
    token_by_path: dict[str, dict] = {}
    large_mode = (sample_limit is None) and (len(manifest_ds) > 50_000)

    shard_root = utter_cache_root / f"{utter_cache_key}.shards"
    checkpoint_rows = 4096  # write one shard roughly every N utterances
    shard_buffer: list[dict] = []

    def _next_shard_index() -> int:
        try:
            existing = [p for p in shard_root.iterdir() if p.is_dir() and p.name.startswith("shard-")]
        except Exception:
            existing = []
        if not existing:
            return 0
        try:
            return 1 + max(int(p.name.split("-")[1]) for p in existing)
        except Exception:
            return len(existing)

    shard_index = _next_shard_index()

    def _flush_shard_buffer() -> None:
        nonlocal shard_index, shard_buffer
        if not shard_buffer:
            return
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_dir = shard_root / f"shard-{shard_index:06d}"
        Dataset.from_list(shard_buffer).save_to_disk(str(shard_dir))
        shard_index += 1
        shard_buffer = []

    def _merge_cache_dir(cache_dir: Path, *, max_to_merge: int | None = None) -> int:
        """Load a cache dir and merge into token_by_path. Returns rows merged."""
        try:
            ds = load_from_disk(str(cache_dir))
            merged = 0
            for row in ds:
                ap = row.get("audio_path")
                if not ap:
                    continue
                if ap not in token_by_path:
                    token_by_path[str(ap)] = dict(row)
                    merged += 1
                    if max_to_merge is not None and merged >= int(max_to_merge):
                        break
            return merged
        except Exception as e:
            logger.warning(f"Failed to load utterance token cache {cache_dir}: {e}")
            return 0

    def _merge_cache_dir_subset(cache_dir: Path, needed: set[str]) -> int:
        """Merge only rows whose audio_path is in `needed`. Mutates `needed` by removing hits."""
        if not needed:
            return 0
        try:
            ds = load_from_disk(str(cache_dir))
            merged = 0
            for row in ds:
                ap = row.get("audio_path")
                if not ap:
                    continue
                ap = str(ap)
                if ap not in needed:
                    continue
                if ap not in token_by_path:
                    token_by_path[ap] = dict(row)
                    merged += 1
                needed.discard(ap)
                if not needed:
                    break
            return merged
        except Exception as e:
            logger.warning(f"Failed to load utterance token cache {cache_dir}: {e}")
            return 0

    # Always load sharded checkpoints first (these represent the real long-run progress).
    merged_shards = 0
    shard_dirs: list[Path] = []
    if shard_root.exists():
        try:
            shard_dirs = sorted([p for p in shard_root.iterdir() if p.is_dir() and p.name.startswith("shard-")])
        except Exception:
            shard_dirs = []
        if shard_dirs:
            if not large_mode:
                logger.info(f"Loading {len(shard_dirs)} cached utterance shards from {shard_root}")
                if subset_needed_paths is not None:
                    needed = set(subset_needed_paths)
                    for sd in shard_dirs:
                        merged_shards += _merge_cache_dir_subset(sd, needed)
                        if not needed:
                            break
                    if needed:
                        logger.info(f"Subset cache load: still missing {len(needed)} paths after scanning shards")
                else:
                    for sd in shard_dirs:
                        merged_shards += _merge_cache_dir(sd, max_to_merge=None)

    # Then load the flat cache (if it exists). This can be small (e.g. earlier --limit runs),
    # but it should never block shard resume.
    merged_flat = 0
    if utter_cache.exists():
        logger.info(f"Loading cached utterance tokens from {utter_cache}")
        if not large_mode:
            if subset_needed_paths is not None:
                needed = set(subset_needed_paths)
                merged_flat = _merge_cache_dir_subset(utter_cache, needed)
            else:
                merged_flat = _merge_cache_dir(utter_cache)
        logger.info(f"Cached utterance token rows loaded: {merged_flat}")
    else:
        # Legacy caches: if user already cached token IDs under an older key, reuse them as a seed.
        if utter_cache_root.exists():
            try:
                legacy_dirs = [
                    p
                    for p in utter_cache_root.iterdir()
                    if p.is_dir() and not p.name.endswith(".shards") and (p / "dataset_info.json").exists()
                ]
            except Exception:
                legacy_dirs = []
            if legacy_dirs:
                logger.info(f"Seeding utterance token cache from {len(legacy_dirs)} existing cache dirs")
                for p in legacy_dirs:
                    _merge_cache_dir(p, max_to_merge=None)

    if shard_dirs or utter_cache.exists():
        logger.info(
            "Utterance token cache resume | shards=%s merged_shards=%s | merged_flat=%s | total_cached=%s",
            len(shard_dirs),
            merged_shards,
            merged_flat,
            len(token_by_path),
        )

    if large_mode:
        # Full-corpus mode: avoid materializing 700k rows into Python.
        # We resume from shards by building a set of cached audio paths, then streaming the manifest.
        cached_paths: set[str] = set()
        if shard_dirs:
            logger.info("Indexing cached shard audio_path values (for resume)...")
            for sd in shard_dirs:
                try:
                    ds = load_from_disk(str(sd))
                    for ap in ds["audio_path"]:
                        cached_paths.add(str(ap))
                except Exception as e:
                    logger.warning(f"Failed to index shard {sd}: {e}")

        logger.info(f"Resume index: cached_paths={len(cached_paths)}")

        def _on_batch(rows: list[dict]) -> None:
            for row in rows:
                ap = str(row.get("audio_path"))
                if ap:
                    cached_paths.add(ap)
                    shard_buffer.append(row)
            if len(shard_buffer) >= checkpoint_rows:
                _flush_shard_buffer()

        # Stream through manifest and tokenize only missing items, chunking to keep memory bounded.
        chunk: list[dict] = []
        for ex in tqdm(manifest_ds, desc="Selecting missing for tokenization", unit="file"):
            ap = str(ex["audio_path"])
            if ap in cached_paths:
                continue
            chunk.append(
                {
                    "audio_path": ap,
                    "text": ex["text"],
                    "speaker": ex["speaker"],
                    "duration": float(ex["duration"]),
                }
            )
            if len(chunk) >= 4096:
                preprocess_samples_batched_ids(
                    samples=chunk,
                    audio_tokenizer=audio_tokenizer,
                    batch_size=tokenizer_batch_size,
                    num_workers=num_loading_workers,
                    on_batch=_on_batch,
                    collect=False,
                )
                chunk = []
        if chunk:
            preprocess_samples_batched_ids(
                samples=chunk,
                audio_tokenizer=audio_tokenizer,
                batch_size=tokenizer_batch_size,
                num_workers=num_loading_workers,
                on_batch=_on_batch,
                collect=False,
            )
        try:
            _flush_shard_buffer()
        except Exception as e:
            logger.warning(f"Failed to flush shard buffer: {e}")

        # Build the training dataset from shards on disk (no giant in-memory lists).
        shard_dirs = sorted([p for p in shard_root.iterdir() if p.is_dir() and p.name.startswith("shard-")]) if shard_root.exists() else []
        if not shard_dirs:
            raise ValueError(f"No utterance token shards found at {shard_root}")

        logger.info(f"Loading utterance token shards for training: {len(shard_dirs)} shards")
        utter_parts = [load_from_disk(str(sd)) for sd in shard_dirs]
        utter_ds = concatenate_datasets(utter_parts)

        # Build a per-speaker reference pool for clone-conditioned training.
        # We keep bounded memory by storing up to `max_keep` examples per speaker.
        by_speaker: dict[str, list[dict]] = defaultdict(list)
        if use_cloning_pairs and float(clone_cross_prob) > 0:
            max_keep = 32
            for row in tqdm(utter_ds, desc="Building speaker ref pool", unit="utt"):
                spk = str(row.get("speaker", ""))
                if not spk:
                    continue
                buf = by_speaker[spk]
                if len(buf) < max_keep:
                    buf.append(row)

            # Stabilize ordering for deterministic reference selection.
            for spk, buf in by_speaker.items():
                try:
                    buf.sort(key=lambda r: str(r.get("audio_path", "")))
                except Exception:
                    pass

        def _select_ref_global_ids(spk: str, audio_path: str | None, self_g_ids: list[int]) -> list[int]:
            """Mix self vs cross-utterance globals deterministically per utterance."""
            if not (use_cloning_pairs and float(clone_cross_prob) > 0):
                return self_g_ids
            buf = by_speaker.get(spk) or []
            if len(buf) < 2:
                return self_g_ids
            ap = str(audio_path or "")
            # Deterministic coin flip per utterance.
            h = hashlib.md5(f"refsel:{ap}:{clone_cross_prob}:{train_objective}:{manifest_key}".encode()).hexdigest()
            u = int(h[:8], 16) / float(0xFFFFFFFF)
            if u >= float(clone_cross_prob):
                return self_g_ids
            # Deterministic reference choice per utterance.
            h2 = hashlib.md5(f"refidx:{ap}:{manifest_key}".encode()).hexdigest()
            idx = int(h2[:8], 16) % len(buf)
            ref = buf[idx]
            if str(ref.get("audio_path", "")) == ap:
                ref = buf[(idx + 1) % len(buf)]
            g = ref.get("global_ids") or self_g_ids
            return list(g)

        # Map utterance tokens -> input_ids/labels in a batched, disk-backed way.
        token_cache: dict[str, int] = {}
        tok_task = "<|task_tts|>"
        tok_start_content = "<|start_content|>"
        tok_end_content = "<|end_content|>"
        tok_start_global = "<|start_global_token|>"
        tok_end_global = "<|end_global_token|>"
        tok_start_sem = "<|start_semantic_token|>"
        tok_end_sem = "<|end_semantic_token|>"
        tok_im_end = "<|im_end|>"
        _ = _require_token_id(tokenizer, tok_task, token_cache)
        _ = _require_token_id(tokenizer, tok_start_content, token_cache)
        _ = _require_token_id(tokenizer, tok_end_content, token_cache)
        start_global_id = _require_token_id(tokenizer, tok_start_global, token_cache)
        end_global_id = _require_token_id(tokenizer, tok_end_global, token_cache)
        start_sem_id = _require_token_id(tokenizer, tok_start_sem, token_cache)
        end_sem_id = _require_token_id(tokenizer, tok_end_sem, token_cache)
        im_end_id = _require_token_id(tokenizer, tok_im_end, token_cache)

        def _map_batch(batch: dict) -> dict:
            out_input_ids = []
            out_attn = []
            out_labels = []
            texts = batch["text"]
            speakers = batch.get("speaker", [""] * len(texts))
            audio_paths = batch.get("audio_path", [""] * len(texts))
            globals_list = batch["global_ids"]
            sem_list = batch["semantic_ids"]
            for text, spk, ap, g_ids, s_ids in zip(texts, speakers, audio_paths, globals_list, sem_list):
                prompt = "".join([tok_task, tok_start_content, str(text), tok_end_content])
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                ref_g_ids = _select_ref_global_ids(str(spk), str(ap), list(g_ids))
                global_token_ids = [
                    _require_token_id(tokenizer, f"<|bicodec_global_{int(i)}|>", token_cache) for i in ref_g_ids
                ]
                semantic_token_ids = [
                    _require_token_id(tokenizer, f"<|bicodec_semantic_{int(i)}|>", token_cache) for i in s_ids
                ]
                if str(train_objective).lower() in {"clone_semantic", "clone_semantic_v1"}:
                    context_ids = (
                        prompt_ids
                        + [start_global_id]
                        + global_token_ids
                        + [end_global_id]
                        + [start_sem_id]
                    )
                    completion_ids = semantic_token_ids + [end_sem_id] + [im_end_id]
                    if len(context_ids) >= max_seq_length:
                        # Skip pathological samples (should be extremely rare).
                        continue
                    max_completion = max_seq_length - len(context_ids)
                    if max_completion <= 2:
                        continue
                    sem_keep = max_completion - 2
                    sem_ids_trunc = semantic_token_ids[:sem_keep]
                    if not sem_ids_trunc:
                        continue
                    completion_ids = sem_ids_trunc + [end_sem_id] + [im_end_id]
                    input_ids = context_ids + completion_ids
                    labels = [-100] * len(context_ids) + completion_ids.copy()
                else:
                    completion_ids = (
                        [start_global_id]
                        + global_token_ids
                        + [end_global_id]
                        + [start_sem_id]
                        + semantic_token_ids
                        + [end_sem_id]
                        + [im_end_id]
                    )
                    input_ids = prompt_ids + completion_ids
                    prompt_len = len(prompt_ids)
                    if len(input_ids) > max_seq_length:
                        if prompt_len >= max_seq_length:
                            prompt_len = max_seq_length
                            input_ids = input_ids[:max_seq_length]
                        else:
                            input_ids = input_ids[:max_seq_length]
                    if prompt_len > len(input_ids):
                        prompt_len = len(input_ids)
                    labels = [-100] * prompt_len + input_ids[prompt_len:].copy()
                out_input_ids.append(input_ids)
                out_attn.append([1] * len(input_ids))
                out_labels.append(labels)
            return {"input_ids": out_input_ids, "attention_mask": out_attn, "labels": out_labels}

        logger.info("Building tokenized training dataset (streaming map)...")
        train_ds = utter_ds.map(
            _map_batch,
            batched=True,
            batch_size=64,
            remove_columns=utter_ds.column_names,
            desc="Tokenizing for training",
        )
        train_ds = train_ds.filter(lambda ex: len(ex["input_ids"]) <= max_seq_length)

        cache_path.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(dataset_cache))
        logger.info(f"Cached dataset to {dataset_cache}")
        return train_ds

    # Small/limit mode: keep simpler in-memory path.
    samples = list(manifest_ds)
    needed_paths = [str(s["audio_path"]) for s in samples]
    missing = [s for s in samples if str(s["audio_path"]) not in token_by_path]

    if missing:
        logger.info(
            f"Tokenizing missing utterances into token IDs: missing={len(missing)}/{len(samples)} (batch_size={tokenizer_batch_size})..."
        )
        start_time = time.time()

        def _on_batch(rows: list[dict]) -> None:
            # Update in-memory map.
            for row in rows:
                ap = str(row.get("audio_path"))
                if ap:
                    token_by_path[ap] = row
                    shard_buffer.append(row)
            # Periodically flush to disk so interrupts keep progress.
            if len(shard_buffer) >= checkpoint_rows:
                _flush_shard_buffer()

        new_utterances, tokenization_errors = preprocess_samples_batched_ids(
            samples=missing,
            audio_tokenizer=audio_tokenizer,
            batch_size=tokenizer_batch_size,
            num_workers=num_loading_workers,
            on_batch=_on_batch,
        )
        elapsed = time.time() - start_time
        rate = len(new_utterances) / elapsed if elapsed > 0 else 0
        logger.info(f"Tokenized {len(new_utterances)} utterances in {elapsed:.1f}s ({rate:.1f} files/sec)")
        if tokenization_errors > 0:
            logger.warning(f"{tokenization_errors} samples failed tokenization")

        # Flush any remaining buffered rows to disk.
        try:
            _flush_shard_buffer()
        except Exception as e:
            logger.warning(f"Failed to flush shard buffer: {e}")

        # Persist updated cache (full map so future runs can reuse any subset).
        # Note: this can be large for full corpus; shards already provide resumability.
        try:
            utter_cache_root.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(list(token_by_path.values())).save_to_disk(str(utter_cache))
            logger.info(f"Cached utterance tokens to {utter_cache} (rows={len(token_by_path)})")
        except Exception as e:
            logger.warning(f"Failed to save utterance token cache to {utter_cache}: {e}")
    else:
        logger.info(f"All utterances already tokenized in cache: {len(samples)}/{len(samples)}")

    utterances: list[dict] = []
    missing_in_map = 0
    for ap in needed_paths:
        row = token_by_path.get(ap)
        if row is None:
            missing_in_map += 1
            continue
        utterances.append(row)

    if missing_in_map:
        logger.warning(f"{missing_in_map} samples missing from utterance token cache after processing")

    # Build training samples (clone-conditioned objective).
    # Mix self vs cross-utterance reference globals (same speaker) deterministically per utterance.
    by_speaker_small: dict[str, list[dict]] = defaultdict(list)
    if use_cloning_pairs and float(clone_cross_prob) > 0:
        for u in utterances:
            spk = str(u.get("speaker", ""))
            if not spk:
                continue
            by_speaker_small[spk].append(u)
        for spk, buf in by_speaker_small.items():
            try:
                buf.sort(key=lambda r: str(r.get("audio_path", "")))
            except Exception:
                pass

    def _select_ref_small(u: dict) -> list[int]:
        self_g = list(u.get("global_ids") or [])
        if not (use_cloning_pairs and float(clone_cross_prob) > 0):
            return self_g
        spk = str(u.get("speaker", ""))
        buf = by_speaker_small.get(spk) or []
        if len(buf) < 2:
            return self_g
        ap = str(u.get("audio_path", ""))
        h = hashlib.md5(f"refsel:{ap}:{clone_cross_prob}:{train_objective}:{manifest_key}".encode()).hexdigest()
        u01 = int(h[:8], 16) / float(0xFFFFFFFF)
        if u01 >= float(clone_cross_prob):
            return self_g
        h2 = hashlib.md5(f"refidx:{ap}:{manifest_key}".encode()).hexdigest()
        idx = int(h2[:8], 16) % len(buf)
        ref = buf[idx]
        if str(ref.get("audio_path", "")) == ap:
            ref = buf[(idx + 1) % len(buf)]
        return list(ref.get("global_ids") or self_g)

    processed_tokens: list[dict] = []
    for u in utterances:
        processed_tokens.append(
            {
                "text": u["text"],
                "speaker": u.get("speaker"),
                "duration": u.get("duration"),
                "global_ids": _select_ref_small(u),
                "semantic_ids": u["semantic_ids"],
            }
        )

    # Build input_ids/labels directly from IDs (semantic-only loss in clone_semantic objective).
    logger.info("Creating tokenized dataset with loss masking (ID-fastpath)...")
    tokenized_samples = tokenize_for_training_ids(
        processed_tokens,
        tokenizer,
        max_seq_length,
        train_objective=train_objective,
    )
    
    # Filter out samples that are still too long after tokenization (shouldn't happen often)
    final_samples = []
    final_filtered = 0
    for sample in tokenized_samples:
        if len(sample["input_ids"]) <= max_seq_length:
            final_samples.append(sample)
        else:
            final_filtered += 1
    
    if final_filtered > 0:
        logger.warning(f"Filtered {final_filtered} samples that exceeded max_seq_length after tokenization")
    
    if not final_samples:
        raise ValueError("No valid samples after tokenization and filtering")
    
    dataset = Dataset.from_list(final_samples)
    logger.info(f"Final dataset size: {len(dataset)} samples")
    
    # Summary of filtering
    total_raw = len(samples)
    total_final = len(dataset)
    retention_rate = (total_final / total_raw * 100) if total_raw > 0 else 0
    logger.info(f"Dataset filtering summary: {total_raw} raw → {total_final} final ({retention_rate:.1f}% retained)")
    
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dataset_cache))
    logger.info(f"Cached dataset to {dataset_cache}")

    return dataset
