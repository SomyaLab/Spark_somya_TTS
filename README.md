# Spark-Somya-TTS

Zero-shot voice cloning TTS for Indic languages, fine-tuned from Spark-TTS-0.5B.

## Features

- **Zero-shot voice cloning**: Clone any voice with just 3-10 seconds of reference audio
- **Indic language support**: Hindi, Kannada, Tamil, Bengali, Gujarati, Telugu, Marathi, and English
- **Text normalization**: Automatic handling of acronyms, currency, numbers, and symbols
- **Long text support**: Automatic chunking with crossfade for seamless long audio generation
- **FastAPI server**: Production-ready REST API

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Spark_somya_TTS

# Install dependencies (using uv)
uv sync
```

### Running the Server

```bash
# Set the BiCodec directory (required)
export SPARK_TTS_CODEC_DIR=/path/to/Spark-TTS-0.5B

# Start the server
uv run python server.py
```

The server will be available at `http://localhost:8000`.

### API Usage

```bash
# Generate speech from text
curl -X POST "http://localhost:8000/Get_Inference" \
  -F "text=नमस्ते, यह एक परीक्षण है।" \
  -F "speaker_wav=@reference_voice.wav" \
  --output output.wav
```

## Project Structure

```
Spark_somya_TTS/
├── server.py                 # FastAPI server
├── main.py                   # CLI entry point
├── pyproject.toml            # Dependencies
├── finetuned_model/          # Finetuned LLM weights
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer files...
├── src/spark_tts/
│   ├── config.py             # Configuration
│   ├── text_processor.py     # Text normalization & chunking
│   ├── data/
│   │   └── tokenizer.py      # Audio tokenization (BiCodec)
│   ├── inference/
│   │   └── generate.py       # Speech generation
│   └── training/
│       └── trainer.py        # Model training
└── scripts/
    └── upload_to_hf.py       # HuggingFace upload script
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SPARK_TTS_MODEL_DIR` | Unified model directory (LLM + BiCodec) | - |
| `SPARK_TTS_LLM_PATH` | Path to finetuned LLM | `finetuned_model` |
| `SPARK_TTS_CODEC_DIR` | Path to BiCodec directory | `Spark-TTS-0.5B` |

### Model Loading

The server supports two configurations:

1. **Unified directory** (recommended for deployment):
   ```bash
   export SPARK_TTS_MODEL_DIR=/path/to/model
   ```
   The directory should contain both LLM files and BiCodec components.

2. **Separate directories** (for development):
   ```bash
   export SPARK_TTS_LLM_PATH=finetuned_model
   export SPARK_TTS_CODEC_DIR=Spark-TTS-0.5B
   ```

## API Reference

### POST /Get_Inference

Generate speech from text using zero-shot voice cloning.

**Parameters:**
- `text` (required): Input text to convert to speech
- `lang` (optional): Language code (auto-detected if not provided)
- `speaker_wav` (required): Reference WAV file for voice cloning

**Response:** WAV audio file (16kHz, mono)

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/Get_Inference",
    params={"text": "नमस्ते, यह एक परीक्षण है।"},
    files={"speaker_wav": open("reference.wav", "rb")}
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### POST /synthesize

Alternative endpoint with additional options.

**Parameters:**
- `text` (required): Input text
- `lang` (optional): Language code
- `speaker_wav` (required): Reference audio
- `use_long_text` (optional, default=true): Enable chunked processing for long texts

### GET /health

Health check endpoint.

**Response:** `{"status": "healthy", "device": "cuda"}`

## Text Processing

The system automatically processes text before synthesis:

1. **Language detection**: Auto-detects script (Devanagari, Tamil, etc.)
2. **Acronym expansion**: `US` → `U S`
3. **Currency conversion**: `₹500` → `paanch sau rupaye`
4. **Symbol replacement**: `%` → `pratishat` (for Hindi)
5. **Number to words**: `100` → `ek sau`

### Long Text Handling

For texts longer than ~300 characters:

1. Text is split at sentence boundaries (।, ॥, ., !, ?)
2. Each chunk targets ~20 seconds of audio
3. Chunks are generated separately
4. Audio is concatenated with crossfade for smooth transitions

## Training

To fine-tune on your own data:

```bash
# Prepare dataset in datasets/ directory
# Structure: datasets/<lang>/<speaker>/<utterances>.wav + metadata.csv

# Run training
uv run python main.py train
```

See `src/spark_tts/config.py` for training configuration options.

## Supported Languages

| Language | Code | Script |
|----------|------|--------|
| Hindi | hi | Devanagari |
| Kannada | kn | Kannada |
| Tamil | ta | Tamil |
| Bengali | bn | Bengali |
| Gujarati | gu | Gujarati |
| Telugu | te | Telugu |
| Marathi | mr | Devanagari |
| English | en | Latin |

## Upload to HuggingFace

To upload the finetuned model to HuggingFace Hub:

```bash
# Login to HuggingFace
huggingface-cli login

# Upload (downloads BiCodec automatically)
uv run python scripts/upload_to_hf.py
```

Environment variables:
- `HF_REPO_NAME`: Repository name (default: `Spark-Somya-TTS-Indic`)
- `HF_PRIVATE`: Set to `true` for private repo

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)
- ~4GB VRAM for inference
- ~16GB VRAM for training

## License

Apache 2.0
