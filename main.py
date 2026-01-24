"""
Spark-TTS: Text-to-Speech training and inference.

Usage:
    python main.py train                           # Train from scratch (two-phase)
    python main.py train --limit 700               # Test with limited samples
    python main.py train outputs/phase1/ckpt-500   # Resume phase 1
    python main.py train outputs/checkpoint-1000   # Resume phase 2
    python main.py train --resume                  # Auto-detect latest checkpoint
    python main.py infer                           # Generate speech with default text
    python main.py infer "text"                    # Generate speech with custom text
    python main.py clone "text" ref_audio.wav      # Zero-shot voice cloning
"""

import sys
import logging
from pathlib import Path
from unsloth import FastModel

from src.spark_tts.config import Config
from src.spark_tts.training.trainer import run_training, setup_logging, load_model
from src.spark_tts.data.tokenizer import AudioTokenizer
from src.spark_tts.inference.generate import generate_speech, generate_speech_clone, save_audio

logger = logging.getLogger("spark_tts")

def run_inference(text: str | None, config: Config):
    """Generate speech from text."""
    setup_logging()
    
    default_text = "सीरिकल्चर रिसर्च अपने छोट आकार आउर संस्कृति में आसानी के कारण आदर्श जीव बन गइल बा"
    
    if not text:
        logger.info(f"No input text provided. Using default:\n{default_text}")
        text = default_text
    
    if Path(config.lora_dir).exists():
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.lora_dir,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            full_finetuning=getattr(config, "full_finetuning", False),
            load_in_4bit=config.load_in_4bit,
            attn_implementation="sdpa",
        )
    else:
        model, tokenizer = load_model(config)
    
    FastModel.for_inference(model)
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))
    
    # Clone-conditioned inference is the default for this repo.
    wav = None
    obj = str(getattr(config, "train_objective", "")).lower()
    if obj in {"clone_semantic", "clone_semantic_v1"}:
        if getattr(config, "default_ref_audio", None) and Path(str(config.default_ref_audio)).exists():
            wav = generate_speech_clone(text, str(config.default_ref_audio), model, tokenizer, audio_tokenizer, config)
        else:
            logger.error("train_objective=%s but default_ref_audio missing; cannot run infer", obj)
            return None
    else:
        # Legacy fallback: allow plain generation if objective supports it.
        if getattr(config, "default_ref_audio", None) and Path(str(config.default_ref_audio)).exists():
            wav = generate_speech_clone(text, str(config.default_ref_audio), model, tokenizer, audio_tokenizer, config)
        else:
            wav = generate_speech(text, model, tokenizer, audio_tokenizer, config)
    
    if wav.size > 0:
        save_audio(wav, "generated_speech.wav", config.sample_rate)
        return "generated_speech.wav"
    
    logger.error("Generation failed")
    return None


def run_clone(text: str | None = None, ref_audio: str | None = None, config: Config = None):
    """Zero-shot voice cloning: generate speech in reference speaker's voice."""
    setup_logging()

    # Default text and reference audio if not provided
    default_text = "जिसमें जीवन के उच्च आध्यात्मिक मूल्य जीवन की निम्नता और भौतिकता के सम्मुख असमर्थ होते महासमर-बंधन प्रतीत होते हैं और हस्तिनापुर का जीवन महाभारत के युद्ध की दिशा ग्रहण करने लगता है।"
    default_ref_audio = "GNR_hi.wav"

    effective_text = text
    effective_ref_audio = ref_audio

    if not text or str(text).strip() == "":
        logger.info(f"No input text provided. Using default:\n{default_text}")
        effective_text = default_text

    if not ref_audio or not Path(ref_audio).exists():
        if not ref_audio:
            logger.info("No reference audio provided. Using default reference audio sample.")
        else:
            logger.warning(f"Reference audio not found: {ref_audio}. Using default reference audio sample.")
        effective_ref_audio = default_ref_audio

    if Path(config.lora_dir).exists():
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.lora_dir,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            full_finetuning=getattr(config, "full_finetuning", False),
            load_in_4bit=config.load_in_4bit,
            attn_implementation="sdpa",
        )
    else:
        model, tokenizer = load_model(config)

    FastModel.for_inference(model)
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))

    wav = generate_speech_clone(effective_text, effective_ref_audio, model, tokenizer, audio_tokenizer, config)

    if wav.size > 0:
        save_audio(wav, "cloned_speech.wav", config.sample_rate)
        return "cloned_speech.wav"

    logger.error("Voice cloning failed")
    return None


def main():
    config = Config()
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "train":
        # Parse options: --resume, --limit N
        i = 1
        while i < len(args):
            if args[i] == "--resume":
                config.resume_from_checkpoint = True
            elif args[i] == "--limit" and i + 1 < len(args):
                config.sample_limit = int(args[i + 1])
                logger.info(f"Sample limit set to {config.sample_limit}")
                i += 1
            elif not args[i].startswith("--"):
                # Assume it's a checkpoint path
                config.resume_from_checkpoint = args[i]
            i += 1
        run_training(config)

    elif cmd == "infer":
        text = args[1] if len(args) > 1 else None
        out_path = run_inference(text, config)
        if out_path:
            print(str(Path(out_path).resolve()))

    elif cmd == "clone":
        # Accept defaults if not enough args
        text = args[1] if len(args) > 1 else None
        ref_audio = args[2] if len(args) > 2 else None
        out_path = run_clone(text, ref_audio, config)
        if out_path:
            print(str(Path(out_path).resolve()))

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
