"""
Spark-Somya-TTS FastAPI Server

Zero-shot voice cloning API with text normalization and long text support.
"""

import torch
import uvicorn
import io
import os
import shutil
import tempfile
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Query, Request
from unsloth import FastLanguageModel

from spark_tts.config import Config
from spark_tts.data.tokenizer import AudioTokenizer
from spark_tts.inference.generate import generate_speech_clone, prepare_wav_for_save
from spark_tts.text_processor import LongTextProcessor

app = FastAPI(
    title="Spark-Somya-TTS API",
    description="Zero-shot voice cloning for Indic languages",
    version="1.0.0",
)

# ==========================================
# 1. MODEL LOADING
# ==========================================

# Model paths - supports two configurations:
# 1. Unified HF repo: MODEL_DIR contains both LLM and BiCodec
# 2. Separate dirs: LLM_PATH for model, CODEC_DIR for BiCodec

MODEL_DIR = os.environ.get("SPARK_TTS_MODEL_DIR", None)

if MODEL_DIR and os.path.isfile(os.path.join(MODEL_DIR, "config.yaml")):
    # Unified HF repo: both LLM and BiCodec in same directory
    LLM_PATH = MODEL_DIR
    CODEC_DIR = MODEL_DIR
    print(f"Loading from unified model directory: {MODEL_DIR}")
else:
    # Separate directories (backward compatible)
    LLM_PATH = os.environ.get("SPARK_TTS_LLM_PATH", "finetuned_model")
    CODEC_DIR = os.environ.get("SPARK_TTS_CODEC_DIR", "Spark-TTS-0.5B")
    
    if not os.path.isfile(os.path.join(CODEC_DIR, "config.yaml")):
        raise FileNotFoundError(
            f"BiCodec config not found: {CODEC_DIR}/config.yaml. "
            "Either:\n"
            "  1. Set SPARK_TTS_MODEL_DIR to a unified HF repo with both LLM and BiCodec\n"
            "  2. Set SPARK_TTS_CODEC_DIR to the BiCodec directory (with config.yaml, BiCodec/, wav2vec2-large-xlsr-53/)"
        )
    print(f"Loading LLM from: {LLM_PATH}")
    print(f"Loading BiCodec from: {CODEC_DIR}")

MAX_SEQ_LENGTH = 2048
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing Spark-Somya-TTS Server...")

# Load LLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LLM_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)
print("LLM loaded and optimized for inference")

# Load Audio Tokenizer (BiCodec)
audio_tokenizer = AudioTokenizer(CODEC_DIR, str(device))
print("Audio tokenizer loaded")

# Server configuration
class ServerConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_dir = CODEC_DIR
        self.max_new_audio_tokens = 2048
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 0.95
        self.repetition_penalty = 1.2

base_config = ServerConfig()

# Long text processor for chunked generation
long_text_processor = LongTextProcessor(
    target_duration=20.0,
    sample_rate=16000,
    crossfade_ms=100,
)

# Threshold for using long text processing (approximate chars for ~20 sec)
LONG_TEXT_THRESHOLD = 300

print("Server ready!")

# ==========================================
# 2. API ENDPOINTS
# ==========================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "device": device}


@app.api_route("/Get_Inference", methods=["GET", "POST"])
async def get_inference(
    request: Request,
    text: str = Query(..., description="Input text to convert to speech."), 
    lang: str = Query(default=None, description="Language code (auto-detected if not provided)."),
    speaker_wav: UploadFile = File(..., description="Reference WAV file for voice cloning.")
):
    """
    Generate speech audio from text using zero-shot voice cloning.
    
    - Automatically normalizes text (acronyms, currency, numbers)
    - For long texts (>300 chars), splits into chunks and applies crossfade
    - Returns WAV audio at 16kHz
    """
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav_path = temp_wav.name
        try:
            # Save uploaded reference audio
            shutil.copyfileobj(speaker_wav.file, temp_wav)
            temp_wav.close()

            print(f"Processing: lang={lang}, text_length={len(text)}")

            # Choose processing method based on text length
            if len(text) > LONG_TEXT_THRESHOLD:
                # Long text: use chunked generation with crossfade
                print(f"Using long text processing ({len(text)} chars)")
                wav = long_text_processor.generate_long_audio(
                    text=text,
                    ref_audio_path=temp_wav_path,
                    model=model,
                    tokenizer=tokenizer,
                    audio_tokenizer=audio_tokenizer,
                    config=base_config,
                    lang=lang,
                )
            else:
                # Short text: direct generation (text normalization is built-in)
                wav = generate_speech_clone(
                    text=text,
                    ref_audio_path=temp_wav_path,
                    model=model,
                    tokenizer=tokenizer,
                    audio_tokenizer=audio_tokenizer,
                    config=base_config,
                )

            # Prepare audio for output
            wav = prepare_wav_for_save(wav, sample_rate=16000)
            
            if len(wav) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate audio"
                )

            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, wav, 16000, format='WAV')
            buffer.seek(0)
            
            return Response(
                content=buffer.read(),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=output.wav"}
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)


@app.post("/synthesize")
async def synthesize(
    text: str = Query(..., description="Input text to convert to speech."),
    lang: str = Query(default=None, description="Language code (auto-detected if not provided)."),
    speaker_wav: UploadFile = File(..., description="Reference WAV file for voice cloning."),
    use_long_text: bool = Query(default=True, description="Enable chunked processing for long texts."),
):
    """
    Alternative endpoint with more options.
    Same as /Get_Inference but with explicit long text control.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav_path = temp_wav.name
        try:
            shutil.copyfileobj(speaker_wav.file, temp_wav)
            temp_wav.close()

            if use_long_text and len(text) > LONG_TEXT_THRESHOLD:
                wav = long_text_processor.generate_long_audio(
                    text=text,
                    ref_audio_path=temp_wav_path,
                    model=model,
                    tokenizer=tokenizer,
                    audio_tokenizer=audio_tokenizer,
                    config=base_config,
                    lang=lang,
                )
            else:
                wav = generate_speech_clone(
                    text=text,
                    ref_audio_path=temp_wav_path,
                    model=model,
                    tokenizer=tokenizer,
                    audio_tokenizer=audio_tokenizer,
                    config=base_config,
                )

            wav = prepare_wav_for_save(wav, sample_rate=16000)
            
            if len(wav) == 0:
                raise HTTPException(status_code=500, detail="Failed to generate audio")

            buffer = io.BytesIO()
            sf.write(buffer, wav, 16000, format='WAV')
            buffer.seek(0)
            
            return Response(
                content=buffer.read(),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=output.wav"}
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
