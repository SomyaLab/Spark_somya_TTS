import torch
from huggingface_hub import snapshot_download
from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

from ..config import Config


def load_model(config: Config) -> tuple:
    """
    Download and load the Spark-TTS model.

    Returns:
        Tuple of (model, tokenizer)
    """
    snapshot_download(config.model_name, local_dir=config.model_dir)

    model, tokenizer = FastModel.from_pretrained(
        model_name=f"{config.model_dir}/LLM",
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        full_finetuning=config.full_finetuning,
        load_in_4bit=config.load_in_4bit,
    )

    return model, tokenizer


def setup_lora(model, config: Config):
    """
    Configure LoRA adapters on the model.

    Args:
        model: The base model
        config: Configuration with LoRA settings

    Returns:
        Model with LoRA adapters
    """
    model = FastModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    return model


def train(
    model,
    tokenizer,
    dataset: Dataset,
    config: Config,
) -> dict:
    """
    Train the model using SFTTrainer.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        dataset: Training dataset
        config: Training configuration

    Returns:
        Training statistics
    """
    sft_config = SFTConfig(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps if config.num_train_epochs is None else -1,
        num_train_epochs=config.num_train_epochs or 1,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=config.output_dir,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
        args=sft_config,
    )

    # Log GPU memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        print(f"GPU: {gpu_stats.name} | Max memory: {max_memory} GB")
        print(f"Reserved before training: {start_memory} GB")

    # Train
    trainer_stats = trainer.train()

    # Log final stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")
        print(f"Peak memory: {used_memory} GB")

    return trainer_stats.metrics


def save_model(
    model,
    tokenizer,
    config: Config,
    merge_16bit: bool = False,
):
    """
    Save the trained model.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: Configuration
        merge_16bit: If True, save merged 16-bit model instead of LoRA adapters
    """
    if merge_16bit:
        model.save_pretrained_merged(
            config.lora_dir,
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        model.save_pretrained(config.lora_dir)
        tokenizer.save_pretrained(config.lora_dir)

    print(f"Model saved to {config.lora_dir}")
