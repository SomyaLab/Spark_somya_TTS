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
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    # Full finetuning on 0.5B is typically stable in ~1e-5..5e-5.
    learning_rate: float = 5e-5
    warmup_steps: int = 1200
    max_steps: int = 16000
    num_train_epochs: Optional[int] = None
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 1000
    optim: str = "adamw_8bit"  # Can set to "adamw_torch" for maximum stability
    # Safety guard: prevents accidentally using a destructive LR for full finetuning.
    unsafe_allow_high_lr: bool = False
    
    # Evaluation / best checkpoint
    # NOTE: This repo uses TRL SFTTrainer (HF Trainer underneath). "Best model" is defined by eval metric.
    do_eval: bool = True
    val_split: float = 0.02  # fraction of dataset reserved for eval (auto-disabled if dataset too small)
    evaluation_strategy: str = "steps"  # "steps" or "no"
    eval_steps: int = 1000
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: int = 2  # keep small number of checkpoints on disk
    
    # Logging (keep simple; rely on HF Trainer logging_steps)
    eval_during_phase1: bool = False  # default: eval only during Phase 2
    max_grad_norm: float = 0.5
    label_smoothing: float = 0
    resume_from_checkpoint: str | bool = False  # Path, True (latest), or False. Default False to start fresh.
    
    # Two-phase training settings (for extended tokenizer)
    embedding_warmup_steps: int = 2500
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
    # Cloning semantic decoding: minimum tokens before allowing stop.
    min_new_semantic_tokens: int = 200
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
    # Training objective:
    # - "clone_semantic": (text + reference globals) -> semantic tokens (recommended for cloning)
    # - any other value falls back to legacy objective (model predicts globals + semantics)
    train_objective: str = "clone_semantic"
    # Probability of using cross-utterance reference globals (same speaker) vs self globals.
    clone_cross_prob: float = 0.8
    
    # Dataset sampling (for testing/debugging)
    sample_limit: int | None = None  # Limit number of samples (None = use all)
    # Audio tokenization is the long pole. Increase on big GPUs (A100 can usually handle 32+).
    tokenizer_batch_size: int = 64  # Batch size for audio tokenization
    num_loading_workers: int = 16 # Parallel audio loading workers (I/O bound; increase on fast disks)

    # =========================
    # Multilingual training plan
    # =========================
    # We use a 2-stage schedule for Indic adaptation:
    #   - Stage 1: Indic-heavy embedding warmup (Phase 1 in trainer)
    #   - Stage 2: full fine-tune (Phase 2 in trainer)
    two_stage: bool = True

    # Dataset language codes are inferred from folder layout: datasets/<lang>/...
    # These defaults cover a typical 10-language Indic setup.
    indic_languages: list[str] = field(
        default_factory=lambda: ["hi", "kn", "te", "bh", "mr", "mai", "mag", "gu", "bn", "hne"]
    )
    base_languages: list[str] = field(default_factory=list)

    # Stage language filters:
    # - None means "use whatever languages exist in the dataset root".
    stage1_languages: list[str] | None = field(
        default_factory=lambda: ["hi", "kn", "te", "bh", "mr", "mai", "mag", "gu", "bn", "hne"]
    )
    stage2_languages: list[str] | None = None

    # Language sampling strategy for Stage 2:
    # - "proportional": keep all selected utterances as-is
    # - "balanced": cap non-Indic share to `max_base_fraction` (when base_languages is non-empty)
    language_sampling: str = "balanced"
    max_base_fraction: float = 0.30
    # Optional hard cap per language (useful to keep training cost bounded).
    max_samples_per_language: int | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
