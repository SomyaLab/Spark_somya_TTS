"""Training utilities for Spark-TTS."""

import logging
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from transformers import TrainerCallback

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
            # Print directly to ensure visibility
            print(f"  → Normalized loss: {normalized_loss:.4f} (raw: {raw_loss:.4f}, GA={ga_steps})")


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


def _build_sft_config(config: Config, output_dir: str, max_steps: int, lr: float, **overrides) -> SFTConfig:
    """Create SFTConfig for training with pre-tokenized dataset."""
    return SFTConfig(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=overrides.get("warmup_steps", config.warmup_steps),
        max_steps=max_steps,
        learning_rate=lr,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type=overrides.get("lr_scheduler_type", config.lr_scheduler_type),
        seed=config.seed,
        output_dir=output_dir,
        save_steps=overrides.get("save_steps", config.save_steps),
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing,
        fp16=False,
        bf16=True,
        logging_dir=f"{output_dir}/logs",
        report_to="tensorboard",
        # Note: dataset is pre-tokenized with input_ids/labels, no text field needed
        max_seq_length=config.max_seq_length,
    )


def _create_trainer(model, tokenizer, dataset: Dataset, config: Config, sft_config: SFTConfig) -> SFTTrainer:
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
        packing=False,
        args=sft_config,
        callbacks=[NormalizedLossCallback()],
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


def train(model, tokenizer, dataset: Dataset, config: Config) -> dict:
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
    
    # ==================== PHASE 1: Embedding Warmup ====================
    if start_phase == 1:
        logger.info("=" * 30 + " PHASE 1: Embedding Warmup " + "=" * 30)
        
        if config.freeze_base_during_warmup:
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
            warmup_steps=50,
            lr_scheduler_type="cosine",
        )
        
        trainer = _create_trainer(model, tokenizer, dataset, config, phase1_config)
        _log_gpu_stats("Before Phase 1: ")
        phase1_stats = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info(f"Phase 1 complete | Loss: {phase1_stats.metrics.get('train_loss', 'N/A')}")
        
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
    )
    
    trainer = _create_trainer(model, tokenizer, dataset, config, phase2_config)
    _log_gpu_stats("Before Phase 2: ")
    phase2_stats = trainer.train(resume_from_checkpoint=checkpoint)
    _log_gpu_stats("After Phase 2: ")
    
    logger.info(f"Phase 2 complete | Loss: {phase2_stats.metrics.get('train_loss', 'N/A')}")
    
    return {
        "phase1": phase1_stats.metrics if phase1_stats else None,
        "phase2": phase2_stats.metrics,
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
    )
    logger.info(f"Dataset: {len(dataset)} samples (pre-tokenized with loss masking)")
    audio_tokenizer.offload_to_cpu()
    
    metrics = train(model, tokenizer, dataset, config)
    
    save_model(model, tokenizer, config)
    return metrics
