"""Training utilities for Spark-TTS."""

import logging
import torch
from pathlib import Path
from dataclasses import replace
from huggingface_hub import snapshot_download
from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from transformers import AutoTokenizer

from ..config import Config
from ..data.tokenizer import AudioTokenizer
from ..data.dataset import load_local_dataset

logger = logging.getLogger("spark_tts")

def setup_logging(level: str = "INFO"):
    """Configure logging once at startup."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Unsloth/Transformers may pre-configure handlers; ensure our logs show up.
    )


def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Find most recent checkpoint in output_dir.
    
    Args:
        output_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint or None if none found
    """
    try:
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        return str(checkpoints[-1]) if checkpoints else None
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing checkpoint names in {output_dir}: {e}")
        return None


def _detect_resume_phase(checkpoint: str | bool, config: Config) -> tuple[str | None, int]:
    """Detect which phase to resume from based on checkpoint path."""
    phase1_dir = f"{config.output_dir}/phase1"
    phase2_dir = config.output_dir
    
    if checkpoint is True:
        phase1_ckpt = _find_latest_checkpoint(phase1_dir)
        phase2_ckpt = _find_latest_checkpoint(phase2_dir)
        
        if phase2_ckpt:
            logger.info(f"Found Phase 2 checkpoint: {phase2_ckpt}")
            return phase2_ckpt, 2
        elif phase1_ckpt:
            logger.info(f"Found Phase 1 checkpoint: {phase1_ckpt}")
            return phase1_ckpt, 1
        else:
            logger.info("No checkpoints found, starting fresh")
            return None, 1
    elif checkpoint:
        if "phase1" in str(checkpoint):
            return str(checkpoint), 1
        else:
            return str(checkpoint), 2
    
    return None, 1


def _get_base_tokenizer_len(config: Config) -> int | None:
    """Best-effort: load base tokenizer length (pre-extension).

    We use this to identify the range of *newly added* tokens in an extended tokenizer.
    Assumption: new tokens were appended (HF `tokenizer.add_tokens`) so new IDs are
    contiguous in [base_len, extended_len).
    """
    base_tok_path = Path(config.model_dir) / "LLM"
    if not base_tok_path.exists():
        # Ensure base model exists locally (needed to load base tokenizer).
        try:
            logger.info(f"Downloading base model {config.model_name} to {config.model_dir} (for base tokenizer)")
            snapshot_download(config.model_name, local_dir=config.model_dir)
        except Exception as e:
            logger.warning(f"Failed to download base model for tokenizer length: {e}")
            return None

    try:
        base_tok = AutoTokenizer.from_pretrained(str(base_tok_path))
        return len(base_tok)
    except Exception as e:
        logger.warning(f"Failed to load base tokenizer from {base_tok_path}: {e}")
        return None


def _mask_embedding_grads_to_new_tokens(
    model,
    tokenizer,
    config: Config,
) -> tuple[torch.utils.hooks.RemovableHandle | None, dict]:
    """Restrict embedding gradient updates to new tokens only (Phase 1 stability).

    This prevents Phase 1 from accidentally training the entire embedding table
    (including audio tokens), which can cause huge gradient norms and instability.
    """
    if not (config.use_extended_model and Path(config.extended_model_dir).exists()):
        return None, {"enabled": False, "reason": "not_using_extended_model"}

    base_len = _get_base_tokenizer_len(config)
    if base_len is None:
        return None, {"enabled": False, "reason": "base_tokenizer_unavailable"}

    ext_len = len(tokenizer)
    if ext_len <= base_len:
        return None, {"enabled": False, "reason": "no_new_tokens_detected", "base_len": base_len, "ext_len": ext_len}

    emb = getattr(model, "get_input_embeddings", None)
    if emb is None or model.get_input_embeddings() is None:
        return None, {"enabled": False, "reason": "no_input_embeddings"}

    emb_weight = model.get_input_embeddings().weight

    # Freeze everything, then train only embedding weights (row-masked to new tokens).
    for p in model.parameters():
        p.requires_grad = False
    emb_weight.requires_grad = True

    # Row mask: 1 for new tokens, 0 otherwise. Keep fp32 for numerical stability.
    # Shape: [vocab, 1] so it broadcasts over hidden dim.
    row_mask = torch.zeros((emb_weight.shape[0], 1), device=emb_weight.device, dtype=torch.float32)
    row_mask[base_len:ext_len] = 1.0

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        # Grad can be fp32 even when params are bf16; multiply mask safely.
        if grad is None:
            return grad
        # Important: do NOT change dtype of grad (PyTorch will error).
        mask = row_mask
        if mask.device != grad.device:
            mask = mask.to(device=grad.device)
        if mask.dtype != grad.dtype:
            mask = mask.to(dtype=grad.dtype)
        return grad * mask

    handle = emb_weight.register_hook(_hook)
    info = {
        "enabled": True,
        "base_len": base_len,
        "ext_len": ext_len,
        "new_tokens": ext_len - base_len,
    }
    return handle, info


def _build_sft_config(config: Config, output_dir: str, max_steps: int, lr: float, **overrides) -> SFTConfig:
    """Create SFTConfig for training with pre-tokenized dataset."""
    # Safety guard: full finetuning is extremely sensitive to LR. If LR is too high,
    # you can get "looks like training works" loss, but completely broken generations.
    effective_lr = lr
    try:
        if (
            getattr(config, "full_finetuning", False)
            and not getattr(config, "unsafe_allow_high_lr", False)
            and lr is not None
            and float(lr) > 5e-5
        ):
            logger.warning(
                "Clamping learning_rate from %.2e to %.2e for stability (set unsafe_allow_high_lr=True to disable).",
                float(lr),
                5e-5,
            )
            effective_lr = 5e-5
    except Exception:
        # If anything goes wrong, fall back to provided LR.
        effective_lr = lr

    use_bf16 = bool(getattr(config, "dtype", None) == torch.bfloat16)
    return SFTConfig(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=overrides.get("warmup_steps", config.warmup_steps),
        max_steps=max_steps,
        learning_rate=effective_lr,
        logging_steps=config.logging_steps,
        optim=getattr(config, "optim", "adamw_8bit"),
        weight_decay=config.weight_decay,
        lr_scheduler_type=overrides.get("lr_scheduler_type", config.lr_scheduler_type),
        seed=config.seed,
        output_dir=output_dir,
        save_steps=overrides.get("save_steps", config.save_steps),
        save_total_limit=overrides.get("save_total_limit", config.save_total_limit),
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing,
        fp16=False,
        bf16=use_bf16,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        eval_strategy=overrides.get("eval_strategy", config.evaluation_strategy),
        eval_steps=overrides.get("eval_steps", config.eval_steps),
        load_best_model_at_end=overrides.get("load_best_model_at_end", config.load_best_model_at_end),
        metric_for_best_model=overrides.get("metric_for_best_model", config.metric_for_best_model),
        greater_is_better=overrides.get("greater_is_better", config.greater_is_better),
        # Note: dataset is pre-tokenized with input_ids/labels, no text field needed
        max_seq_length=config.max_seq_length,
    )


def _create_trainer(
    model,
    tokenizer,
    dataset: Dataset,
    config: Config,
    sft_config: SFTConfig,
    eval_dataset: Dataset | None = None,
) -> SFTTrainer:
    """Create SFTTrainer for fine-tuning.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer instance
        dataset: Pre-tokenized training dataset
        config: Training configuration
        sft_config: SFTConfig with training hyperparameters
        
    Returns:
        Configured SFTTrainer instance
    """
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        packing=False,
        args=sft_config,
    )


def _log_gpu_stats(prefix: str = "") -> None:
    """Log GPU memory stats.
    
    Args:
        prefix: Optional prefix string for log message
    """
    if not torch.cuda.is_available():
        return
    try:
        gpu = torch.cuda.get_device_properties(0)
        mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        total_mem = round(gpu.total_memory / 1024**3, 2)
        logger.info(f"{prefix}GPU: {gpu.name} | Memory: {mem}/{total_mem} GB")
    except Exception as e:
        logger.warning(f"Failed to log GPU stats: {e}")


def load_model(config: Config) -> tuple:
    """Load model - extended or base depending on config.
    
    Args:
        config: Configuration with model paths and settings
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        FileNotFoundError: If model directory doesn't exist
        RuntimeError: If model loading fails
    """
    try:
        if config.use_extended_model and Path(config.extended_model_dir).exists():
            logger.info(f"Loading extended model from {config.extended_model_dir}")
            model_path = config.extended_model_dir
        else:
            # Prefer an already-downloaded local base model if present.
            model_path = f"{config.model_dir}/LLM"
            if not Path(model_path).exists():
                # Optional local fallback to avoid lengthy downloads in environments where a
                # compatible model is already available on disk.
                fallback_dir = Path("finetuned_tokenize_bench")
                looks_like_model = fallback_dir.exists() and (fallback_dir / "config.json").exists()
                if config.model_name == "unsloth/Spark-TTS-0.5B" and looks_like_model:
                    logger.info(f"Using local model fallback at {fallback_dir} (skipping download)")
                    model_path = str(fallback_dir)
                else:
                    logger.info(f"Downloading model {config.model_name} to {config.model_dir}")
                    snapshot_download(config.model_name, local_dir=config.model_dir)
                    model_path = f"{config.model_dir}/LLM"
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            full_finetuning=config.full_finetuning,
            load_in_4bit=config.load_in_4bit,
            attn_implementation="sdpa",
        )
        
        logger.info(f"Vocabulary size: {len(tokenizer)}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


def load_tokenizer_only(config: Config):
    """
    Load just the HF tokenizer (no LLM weights).

    This is used to build the pre-tokenized dataset (input_ids/labels) before loading
    the large LLM onto GPU, so audio tokenization can run with more free VRAM.
    """
    try:
        if config.use_extended_model and Path(config.extended_model_dir).exists():
            tok_path = config.extended_model_dir
        else:
            tok_path = f"{config.model_dir}/LLM"
            if not Path(tok_path).exists():
                # Mirror load_model() fallback
                fallback_dir = Path("finetuned_tokenize_bench")
                looks_like_model = fallback_dir.exists() and (fallback_dir / "tokenizer.json").exists()
                if config.model_name == "unsloth/Spark-TTS-0.5B" and looks_like_model:
                    tok_path = str(fallback_dir)
                else:
                    snapshot_download(config.model_name, local_dir=config.model_dir)
                    tok_path = f"{config.model_dir}/LLM"
        tok = AutoTokenizer.from_pretrained(str(tok_path))
        logger.info(f"Loaded tokenizer only from {tok_path} | vocab={len(tok)}")
        return tok
    except Exception as e:
        logger.error(f"Failed to load tokenizer-only: {e}")
        raise RuntimeError(f"Tokenizer-only load failed: {e}") from e


def setup_lora(model, config: Config):
    """Configure LoRA adapters.
    
    Args:
        model: Base model to add LoRA to
        config: Configuration with LoRA parameters
        
    Returns:
        Model with LoRA adapters configured
    """
    try:
        return FastModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
    except Exception as e:
        logger.error(f"Failed to setup LoRA: {e}")
        raise


def save_model(model, tokenizer, config: Config, merge: bool = False) -> None:
    """Save model to lora_dir.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        config: Configuration with save path
        merge: If True, merge LoRA weights into base model
    """
    try:
        if merge:
            model.save_pretrained_merged(config.lora_dir, tokenizer, save_method="merged_16bit")
        else:
            model.save_pretrained(config.lora_dir)
            tokenizer.save_pretrained(config.lora_dir)
        logger.info(f"Model saved to {config.lora_dir}")
    except Exception as e:
        logger.error(f"Failed to save model to {config.lora_dir}: {e}")
        raise


def train(
    model,
    tokenizer,
    dataset: Dataset,
    config: Config,
    eval_dataset: Dataset | None = None,
    *,
    phase1_dataset: Dataset | None = None,
    phase2_dataset: Dataset | None = None,
) -> dict:
    """Two-phase training pipeline with auto-resume.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer instance
        dataset: Pre-tokenized training dataset
        config: Training configuration
        
    Returns:
        Dict with training metrics from both phases
        
    Raises:
        ValueError: If dataset is empty
    """
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")
    
    checkpoint, start_phase = _detect_resume_phase(config.resume_from_checkpoint, config)
    
    phase1_stats = None
    embedding_grad_hook = None
    
    # ==================== PHASE 1: Embedding Warmup ====================
    if start_phase == 1:
        logger.info("=" * 30 + " PHASE 1: Embedding Warmup " + "=" * 30)
        
        if config.freeze_base_during_warmup:
            if getattr(config, "phase1_train_new_tokens_only", True):
                embedding_grad_hook, info = _mask_embedding_grads_to_new_tokens(model, tokenizer, config)
                if info.get("enabled"):
                    logger.info(
                        "Phase 1: training only NEW token embeddings | base_vocab=%s | extended_vocab=%s | new_tokens=%s",
                        info["base_len"],
                        info["ext_len"],
                        info["new_tokens"],
                    )
                else:
                    logger.info(f"Phase 1 new-token-only mode disabled/fallback: {info.get('reason')}")
                    frozen = 0
                    for name, param in model.named_parameters():
                        if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                            param.requires_grad = False
                            frozen += 1
                    logger.info(f"Frozen {frozen} params, training embeddings only")
            else:
                frozen = 0
                for name, param in model.named_parameters():
                    if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                        param.requires_grad = False
                        frozen += 1
                logger.info(f"Frozen {frozen} params, training embeddings only")
        
        phase1_config = _build_sft_config(
            config,
            output_dir=f"{config.output_dir}/phase1",
            max_steps=config.embedding_warmup_steps,
            lr=config.embedding_warmup_lr,
            warmup_steps=400,
            lr_scheduler_type="cosine",
            # Default: keep Phase 1 simple/fast (no eval/best-model selection)
            eval_strategy="steps" if (config.eval_during_phase1 and eval_dataset is not None and config.do_eval) else "no",
            load_best_model_at_end=False,
        )
        
        trainer = _create_trainer(
            model,
            tokenizer,
            (phase1_dataset if phase1_dataset is not None else dataset),
            config,
            phase1_config,
            eval_dataset=(eval_dataset if (config.eval_during_phase1 and eval_dataset is not None and config.do_eval) else None),
        )
        _log_gpu_stats("Before Phase 1: ")
        phase1_stats = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info(f"Phase 1 complete | Loss: {phase1_stats.metrics.get('train_loss', 'N/A')}")

        if embedding_grad_hook is not None:
            try:
                embedding_grad_hook.remove()
            except Exception as e:
                logger.warning(f"Failed to remove embedding grad hook: {e}")
            embedding_grad_hook = None
        
        checkpoint = None
    else:
        logger.info("Skipping Phase 1 (resuming Phase 2)")
    
    # ==================== PHASE 2: Full Fine-tuning ====================
    logger.info("=" * 30 + " PHASE 2: Full Fine-tuning " + "=" * 30)
    
    for param in model.parameters():
        param.requires_grad = True
    
    phase2_config = _build_sft_config(
        config,
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        lr=config.learning_rate,
        eval_strategy=("steps" if (eval_dataset is not None and config.do_eval) else "no"),
        load_best_model_at_end=(config.load_best_model_at_end and eval_dataset is not None and config.do_eval),
    )
    
    trainer = _create_trainer(
        model,
        tokenizer,
        (phase2_dataset if phase2_dataset is not None else dataset),
        config,
        phase2_config,
        eval_dataset=(eval_dataset if (eval_dataset is not None and config.do_eval) else None),
    )
    _log_gpu_stats("Before Phase 2: ")
    phase2_stats = trainer.train(resume_from_checkpoint=checkpoint)
    _log_gpu_stats("After Phase 2: ")
    
    logger.info(f"Phase 2 complete | Loss: {phase2_stats.metrics.get('train_loss', 'N/A')}")
    
    return {
        "phase1": phase1_stats.metrics if phase1_stats else None,
        "phase2": phase2_stats.metrics,
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
    }


def run_training(config: Config) -> dict:
    """Complete training pipeline - load, setup, train, save."""
    setup_logging()

    # If user requested a small sample_limit, run a short smoke test by default.
    # This keeps `python main.py train --limit 100` usable for verification.
    run_cfg = config
    try:
        is_mini = (config.sample_limit is not None) and (int(config.sample_limit) > 0)
    except Exception:
        is_mini = False
    if is_mini:
        run_cfg = replace(
            config,
            do_eval=False,
            val_split=0.0,
            evaluation_strategy="no",
            load_best_model_at_end=False,
            max_steps=min(int(getattr(config, "max_steps", 60)), 60),
            warmup_steps=min(int(getattr(config, "warmup_steps", 5)), 5),
            save_steps=min(int(getattr(config, "save_steps", 50)), 50),
            embedding_warmup_steps=min(int(getattr(config, "embedding_warmup_steps", 20)), 20),
        )
        logger.info(
            "Mini-run mode enabled (sample_limit=%s) | max_steps=%s | embedding_warmup_steps=%s",
            getattr(config, "sample_limit", None),
            run_cfg.max_steps,
            run_cfg.embedding_warmup_steps,
        )

    # 1) Build dataset first (tokenization is the long pole).
    #    Load only the HF tokenizer, not the full LLM weights.
    logger.info(f"Preparing dataset from {run_cfg.data_dir} (LLM not loaded yet)...")
    tokenizer = load_tokenizer_only(run_cfg)
    audio_tokenizer = AudioTokenizer(run_cfg.model_dir, str(run_cfg.device))
    dataset_stage1 = None
    dataset_stage2 = None

    if getattr(run_cfg, "two_stage", False):
        logger.info("Building Stage 1 dataset (Indic warmup subset)...")
        try:
            dataset_stage1 = load_local_dataset(
                run_cfg.data_dir,
                audio_tokenizer,
                tokenizer=tokenizer,  # For pre-tokenization with loss masking
                min_duration=run_cfg.min_audio_duration,
                max_duration=run_cfg.max_audio_duration,
                max_seq_length=run_cfg.max_seq_length,
                model_dir=run_cfg.model_dir,
                sample_rate=run_cfg.sample_rate,
                use_cloning_pairs=run_cfg.use_cloning_pairs,
                train_objective=getattr(run_cfg, "train_objective", "clone_semantic"),
                clone_cross_prob=float(getattr(run_cfg, "clone_cross_prob", 0.8)),
                sample_limit=run_cfg.sample_limit,
                languages=getattr(run_cfg, "stage1_languages", None),
                language_sampling="proportional",
                base_languages=getattr(run_cfg, "base_languages", None),
                max_base_fraction=float(getattr(run_cfg, "max_base_fraction", 0.30)),
                max_samples_per_language=getattr(run_cfg, "max_samples_per_language", None),
                seed=int(getattr(run_cfg, "seed", 42)),
                tokenizer_batch_size=run_cfg.tokenizer_batch_size,
                num_loading_workers=getattr(run_cfg, "num_loading_workers", 4),
            )
            logger.info(f"Stage 1 dataset: {len(dataset_stage1)} samples")
        except Exception as e:
            dataset_stage1 = None
            logger.warning(f"Stage 1 dataset build failed; proceeding with Stage 2 only: {e}")

    logger.info("Building Stage 2 dataset (multilingual)...")
    dataset_stage2 = load_local_dataset(
        run_cfg.data_dir,
        audio_tokenizer,
        tokenizer=tokenizer,  # For pre-tokenization with loss masking
        min_duration=run_cfg.min_audio_duration,
        max_duration=run_cfg.max_audio_duration,
        max_seq_length=run_cfg.max_seq_length,
        model_dir=run_cfg.model_dir,
        sample_rate=run_cfg.sample_rate,
        use_cloning_pairs=run_cfg.use_cloning_pairs,
        train_objective=getattr(run_cfg, "train_objective", "clone_semantic"),
        clone_cross_prob=float(getattr(run_cfg, "clone_cross_prob", 0.8)),
        sample_limit=run_cfg.sample_limit,
        languages=getattr(run_cfg, "stage2_languages", None),
        language_sampling=getattr(run_cfg, "language_sampling", "proportional"),
        base_languages=getattr(run_cfg, "base_languages", None),
        max_base_fraction=float(getattr(run_cfg, "max_base_fraction", 0.30)),
        max_samples_per_language=getattr(run_cfg, "max_samples_per_language", None),
        seed=int(getattr(run_cfg, "seed", 42)),
        tokenizer_batch_size=run_cfg.tokenizer_batch_size,
        num_loading_workers=getattr(run_cfg, "num_loading_workers", 4),
    )
    logger.info(f"Stage 2 dataset: {len(dataset_stage2)} samples")
    audio_tokenizer.offload_to_cpu()

    # 2) Now load the LLM and start training.
    logger.info("Loading model...")
    model, model_tokenizer = load_model(run_cfg)
    # Ensure we train with the tokenizer paired with the model (should match vocab).
    tokenizer = model_tokenizer
    model = setup_lora(model, run_cfg)
    
    # Create evaluation split (optional) on Stage 2 dataset. Best-model selection requires eval.
    eval_dataset = None
    train_dataset = dataset_stage2
    if run_cfg.do_eval and run_cfg.val_split and run_cfg.val_split > 0:
        try:
            n = len(dataset_stage2)
            if n >= 2:
                # Ensure at least 1 sample in eval, but never all samples.
                val_size = int(round(n * float(run_cfg.val_split)))
                val_size = max(1, min(val_size, n - 1))
                split = dataset_stage2.train_test_split(test_size=val_size, seed=run_cfg.seed, shuffle=True)
                train_dataset = split["train"]
                eval_dataset = split["test"]
                logger.info(f"Eval enabled | train={len(train_dataset)} | eval={len(eval_dataset)} | val_split={run_cfg.val_split}")
            else:
                logger.warning("Eval disabled (dataset too small to split)")
        except Exception as e:
            logger.warning(f"Eval disabled (failed to create split): {e}")
            train_dataset = dataset_stage2
            eval_dataset = None
    else:
        logger.info("Eval disabled (do_eval=False or val_split<=0)")
    
    metrics = train(
        model,
        tokenizer,
        train_dataset,
        run_cfg,
        eval_dataset=eval_dataset,
        phase1_dataset=dataset_stage1,
        phase2_dataset=train_dataset,
    )
    if metrics.get("best_model_checkpoint"):
        logger.info(f"Best checkpoint: {metrics['best_model_checkpoint']} (best_metric={metrics.get('best_metric')})")
    
    save_model(model, tokenizer, run_cfg)
    return metrics
