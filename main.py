"""
Spark-TTS: Text-to-Speech training and inference.

Usage:
    python main.py train                           # Train from scratch (two-phase)
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
    
    default_text = "'ಮಹಾಭಾರತ' ಒಂದು ಮಹತ್ತರ ಕೃತಿ ಆಗಿದ್ದು, ಅದು ಭಾರತೀಯ ಜೀವನ, ಚಿಂತನೆ, ತತ್ತ್ವಶಾಸ್ತ್ರ ಹಾಗೂ ವರ್ತನೆಯನ್ನು ಸ್ಪಷ್ಟ ರೂಪದಲ್ಲಿ ಪ್ರದರ್ಶಿಸುತ್ತದೆ."
    
    if not text:
        logger.info(f"No input text provided. Using default:\n{default_text}")
        text = default_text
    
    if Path(config.lora_dir).exists():
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.lora_dir,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
        )
    else:
        model, tokenizer = load_model(config)
    
    FastModel.for_inference(model)
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))
    
    wav = generate_speech(text, model, tokenizer, audio_tokenizer, config)
    
    if wav.size > 0:
        save_audio(wav, "generated_speech1.wav", config.sample_rate)
        return "generated_speech1.wav"
    
    logger.error("Generation failed")
    return None


def run_clone(text: str, ref_audio: str, config: Config):
    """Zero-shot voice cloning: generate speech in reference speaker's voice."""
    setup_logging()
    
    if not Path(ref_audio).exists():
        logger.error(f"Reference audio not found: {ref_audio}")
        return None
    
    if Path(config.lora_dir).exists():
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.lora_dir,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
        )
    else:
        model, tokenizer = load_model(config)
    
    FastModel.for_inference(model)
    audio_tokenizer = AudioTokenizer(config.model_dir, str(config.device))
    
    wav = generate_speech_clone(text, ref_audio, model, tokenizer, audio_tokenizer, config)
    
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
        # Parse checkpoint: --resume (auto) or explicit path
        if len(args) > 1:
            if args[1] == "--resume":
                config.resume_from_checkpoint = True
            else:
                config.resume_from_checkpoint = args[1]
        run_training(config)
    
    elif cmd == "infer":
        text = args[1] if len(args) > 1 else None
        run_inference(text, config)
    
    elif cmd == "clone":
        if len(args) < 3:
            print("Usage: python main.py clone 'text' ref_audio.wav")
            return
        run_clone(args[1], args[2], config)
    
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
