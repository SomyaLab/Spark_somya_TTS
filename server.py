import torch
import uvicorn
import io
import os
import shutil
import tempfile
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Query, Request
from unsloth import FastLanguageModel

# --- IMPORTS FROM YOUR REPO ---
from spark_tts.config import Config
from spark_tts.data.tokenizer import AudioTokenizer
from spark_tts.inference.generate import generate_speech_clone

app = FastAPI(title="Voice Tech For All API")

# ==========================================
# 1. OPTIMIZED MODEL LOADING (Unsloth)
# ==========================================
# LLM: finetuned model (inference entry point, same as main.py infer/clone when lora_dir exists)
LLM_PATH = "finetuned_model"
# BiCodec: must contain config.yaml, BiCodec, wav2vec2-large-xlsr-53 (same as main.py config.model_dir for AudioTokenizer)
CODEC_DIR = os.environ.get("SPARK_TTS_CODEC_DIR", "Spark-TTS-0.5B")
if not os.path.isfile(os.path.join(CODEC_DIR, "config.yaml")):
    raise FileNotFoundError(
        f"BiCodec config not found: {CODEC_DIR}/config.yaml. "
        "Set SPARK_TTS_CODEC_DIR to the base Spark-TTS dir (e.g. pretrained_models/Spark-TTS-0.5B) that has config.yaml, BiCodec, wav2vec2-large-xlsr-53."
    )

MAX_SEQ_LENGTH = 2048
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Initializing Unsloth Optimized Server...")

# Load LLM from finetuned_model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LLM_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)
print("✅ Unsloth Inference Engine Enabled")

# Load Audio Tokenizer from codec dir (BiCodec needs config.yaml etc.; separate from LLM)
audio_tokenizer = AudioTokenizer(CODEC_DIR, str(device))

# Base Config (model_dir for codec; device comes from Config's @property)
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

# ==========================================
# 2. API ENDPOINT IMPLEMENTATION
# ==========================================

@app.api_route("/Get_Inference", methods=["GET", "POST"])
async def get_inference(
    request: Request,
    text: str = Query(..., description="The input text to be converted into speech."), 
    lang: str = Query(default="hi", description="Language of input text (e.g., hi, kn). Optional."),
    speaker_wav: UploadFile = File(..., description="Reference WAV file for speaker voice.")
):
    """
    Generates speech audio from the provided text using the specified language 
    and speaker reference[cite: 5].
    """
    
    # Create a temporary file to store the uploaded WAV
    # (Spark-TTS requires a file path for librosa.load)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        try:
            # Save uploaded file to disk
            shutil.copyfileobj(speaker_wav.file, temp_wav)
            temp_wav_path = temp_wav.name
            temp_wav.close() # Close file handle so librosa can open it

            # Note on 'lang': The current Spark-TTS inference script is language-agnostic 
            # (zero-shot cloning). We pass 'text' and 'audio', but 'lang' is accepted 
            # to fulfill the API spec.
            
            print(f"Processing Request: Lang={lang}, Text Length={len(text)}")

            # Run Inference
            wav = generate_speech_clone(
                text=text,
                ref_audio_path=temp_wav_path,
                model=model,
                tokenizer=tokenizer,
                audio_tokenizer=audio_tokenizer,
                config=base_config
            )

            # Convert to WAV bytes for response
            buffer = io.BytesIO()
            sf.write(buffer, wav, 16000, format='WAV')
            buffer.seek(0)
            
            # Return as audio/wav [cite: 10]
            return Response(content=buffer.read(), media_type="audio/wav")

        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # Cleanup: Delete the temp file
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

if __name__ == "__main__":
    # Host on 0.0.0.0 to be accessible publicly
    uvicorn.run(app, host="0.0.0.0", port=8000)
    