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
    data_dir: str = "dataset"
    
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
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4  # Lower LR for full finetuning (10x lower than LoRA)
    warmup_steps: int = 300
    max_steps: int = 12000
    num_train_epochs: Optional[int] = None
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine_with_restarts"
    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 500
    max_grad_norm: float = 2.0
    label_smoothing: float = 0.001
    resume_from_checkpoint: str | bool = False  # Path, True (latest), or False. Default False to start fresh.
    
    # Two-phase training settings (for extended tokenizer)
    embedding_warmup_steps: int = 2000
    embedding_warmup_lr: float = 1e-4  # Lower for stable embedding warmup
    freeze_base_during_warmup: bool = True

    # Inference settings
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    max_new_audio_tokens: int = 2048

    # Audio settings
    sample_rate: int = 16000
    min_audio_duration: float = 0.5   # Minimum audio duration in seconds
    max_audio_duration: float = 20.0  # Maximum audio duration in seconds
    
    # Zero-shot cloning training (cross-utterance pairs)
    use_cloning_pairs: bool = True  # Train with ref audio from different utterance

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
