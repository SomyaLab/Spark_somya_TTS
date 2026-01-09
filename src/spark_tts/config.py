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
    lora_dir: str = "lora_model"
    data_dir: str = "data/IISc_SYSPIN_Data"

    # Model settings
    max_seq_length: int = 2048
    dtype: torch.dtype = torch.float32
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

    # Training settings
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    max_steps: int = 60
    num_train_epochs: Optional[int] = None
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    logging_steps: int = 1

    # Inference settings
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 1.0
    max_new_audio_tokens: int = 2048

    # Audio settings
    sample_rate: int = 16000

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
