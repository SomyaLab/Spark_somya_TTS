from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Config:
    """Central configuration for Spark-TTS training and inference."""

    # Paths
    model_name: str = "unsloth/Spark-TTS-0.5B"
    model_dir: str = "Spark-TTS-0.5B"
    output_dir: str = "outputs"
    lora_dir: str = "finetuned_model"
    data_dir: str = "datasets"
    
    # Extended model paths (for Indic tokenizer extension)
    extended_model_dir: str = "extended_model"
    use_extended_model: bool = True

    # Model settings
    max_seq_length: int = 2048
    dtype: torch.dtype = torch.bfloat16  # bf16 matches training precision, reduces memory
    load_in_4bit: bool = False
    full_finetuning: bool = True

    # LoRA settings
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # Training settings (optimized for full finetuning stability)
    # NOTE: These defaults are chosen to *avoid model collapse* in full finetuning.
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    # Full finetuning on 0.5B is typically stable in ~1e-5..5e-5.
    learning_rate: float = 4e-5
    warmup_steps: int = 700
    max_steps: int = 6000
    num_train_epochs: Optional[int] = None
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine_with_restarts"
    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 1000
    optim: str = "adamw_8bit"  # Can set to "adamw_torch" for maximum stability
    # Safety guard: prevents accidentally using a destructive LR for full finetuning.
    unsafe_allow_high_lr: bool = False
    
    # Evaluation / best checkpoint
    # NOTE: This repo uses TRL SFTTrainer (HF Trainer underneath). "Best model" is defined by eval metric.
    do_eval: bool = True
    val_split: float = 0.02  # fraction of dataset reserved for eval (auto-disabled if dataset too small)
    evaluation_strategy: str = "steps"  # "steps" or "no"
    eval_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: int = 2  # keep small number of checkpoints on disk
    
    # Console logging (doesn't affect training, just prettifies prints)
    pretty_log_steps: int = 10  # print one concise line every N logging events (0 disables)
    eval_during_phase1: bool = False  # default: eval only during Phase 2
    max_grad_norm: float = 0.5
    label_smoothing: float = 0.0
    resume_from_checkpoint: str | bool = False  # Path, True (latest), or False. Default False to start fresh.
    
    # Two-phase training settings (for extended tokenizer)
    embedding_warmup_steps: int = 2200
    # Phase 1 is the most fragile; keep LR conservative.
    embedding_warmup_lr: float = 5e-5
    freeze_base_during_warmup: bool = True
    # If using `extended_model`, train only newly-added token rows in embeddings during Phase 1.
    phase1_train_new_tokens_only: bool = True

    # Inference settings
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95
    max_new_audio_tokens: int = 2048
    repetition_penalty: float = 1.1
    # For plain `infer`, we need some speaker identity. If not provided, the model has to "invent"
    # global tokens, which often yields poor or silent audio. We therefore use a default reference
    # audio to extract global tokens (same mechanism as clone).
    default_ref_audio: str = "GNR_hi.wav"

    # Audio settings
    sample_rate: int = 16000
    min_audio_duration: float = 0.05   # Minimum audio duration in seconds
    max_audio_duration: float = 30.0  # Maximum audio duration in seconds
    
    # Zero-shot cloning training (cross-utterance pairs)
    use_cloning_pairs: bool = True  # Train with ref audio from different utterance
    
    # Dataset sampling (for testing/debugging)
    sample_limit: int | None = None  # Limit number of samples (None = use all)
    tokenizer_batch_size: int = 16  # Batch size for audio tokenization

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
