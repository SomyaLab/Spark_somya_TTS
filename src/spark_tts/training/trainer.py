"""Training utilities for Spark-TTS."""

import logging
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from transformers import TrainerCallback, AutoTokenizer

from ..config import Config
from ..data.tokenizer import AudioTokenizer
from ..data.dataset import load_local_dataset

logger = logging.getLogger("spark_tts")


class NormalizedLossCallback(TrainerCallback):
    """Log normalized loss (loss / gradient_accumulation_steps) for easier reading."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            ga_steps = args.gradient_accumulation_steps
            raw_loss = logs["loss"]
            normalized_loss = raw_loss / ga_steps
            logs["loss_normalized"] = normalized_loss


class PrettyLogsCallback(TrainerCallback):
    """Concise, readable console logs (doesn't affect training)."""
    
    def __init__(self, every_n_steps: int = 10):
        self.every_n_steps = max(int(every_n_steps), 0)
    
    def _should_print(self, state) -> bool:
        if self.every_n_steps <= 0:
            return False
        # Print at step 1 and then every N steps to avoid spamming
        return state.global_step == 1 or (state.global_step % self.every_n_steps == 0)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self._should_print(state):
            return
        # HF typically provides: loss, learning_rate, grad_norm, epoch, step, etc.
        loss = logs.get("loss")
        loss_norm = logs.get("loss_normalized")
        lr = logs.get("learning_rate")
        grad_norm = logs.get("grad_norm")
        
        parts = [f"step={state.global_step}"]
        if loss is not None:
            if loss_norm is not None:
                parts.append(f"loss={loss:.4f} (norm={loss_norm:.4f})")
            else:
                parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.2f}")
        
        logger.info(" | ".join(parts))
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        msg = f"eval | step={state.global_step} | eval_loss={eval_loss:.4f}"
        if getattr(state, "best_metric", None) is not None:
            # best_metric corresponds to metric_for_best_model
            msg += f" | best={state.best_metric:.4f}"
        if getattr(state, "best_model_checkpoint", None):
            msg += f" | best_ckpt={Path(state.best_model_checkpoint).name}"
        logger.info(msg)


def setup_logging(level: str = "INFO"):
    """Configure logging once at startup."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
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
        bf16=True,
        logging_dir=f"{output_dir}/logs",
        report_to="tensorboard",
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
        callbacks=[
            NormalizedLossCallback(),
            PrettyLogsCallback(every_n_steps=config.pretty_log_steps),
        ],
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
            attn_implementation="flash_attention_2",
        )
        
        logger.info(f"Vocabulary size: {len(tokenizer)}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


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
            dataset,
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
        dataset,
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
    
    logger.info("Loading model...")
    model, tokenizer = load_model(config)
    model = setup_lora(model, config)
    
    logger.info(f"Loading dataset from {config.data_dir}")
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))
    dataset = load_local_dataset(
        config.data_dir,
        audio_tokenizer,
        tokenizer=tokenizer,  # For pre-tokenization with loss masking
        min_duration=config.min_audio_duration,
        max_duration=config.max_audio_duration,
        max_seq_length=config.max_seq_length,
        model_dir=config.model_dir,
        sample_rate=config.sample_rate,
        use_cloning_pairs=config.use_cloning_pairs,
        sample_limit=config.sample_limit,
        tokenizer_batch_size=config.tokenizer_batch_size,
    )
    logger.info(f"Dataset: {len(dataset)} samples (pre-tokenized with loss masking)")
    audio_tokenizer.offload_to_cpu()
    
    # Create evaluation split (optional). Best-model selection requires eval.
    eval_dataset = None
    train_dataset = dataset
    if config.do_eval and config.val_split and config.val_split > 0:
        try:
            n = len(dataset)
            if n >= 2:
                # Ensure at least 1 sample in eval, but never all samples.
                val_size = int(round(n * float(config.val_split)))
                val_size = max(1, min(val_size, n - 1))
                split = dataset.train_test_split(test_size=val_size, seed=config.seed, shuffle=True)
                train_dataset = split["train"]
                eval_dataset = split["test"]
                logger.info(f"Eval enabled | train={len(train_dataset)} | eval={len(eval_dataset)} | val_split={config.val_split}")
            else:
                logger.warning("Eval disabled (dataset too small to split)")
        except Exception as e:
            logger.warning(f"Eval disabled (failed to create split): {e}")
            train_dataset = dataset
            eval_dataset = None
    else:
        logger.info("Eval disabled (do_eval=False or val_split<=0)")
    
    metrics = train(model, tokenizer, train_dataset, config, eval_dataset=eval_dataset)
    if metrics.get("best_model_checkpoint"):
        logger.info(f"Best checkpoint: {metrics['best_model_checkpoint']} (best_metric={metrics.get('best_metric')})")
    
    save_model(model, tokenizer, config)
    return metrics
