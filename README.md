# Spark-Somya-TTS

Fine-tune Spark-TTS 0.5B on custom voice data using LoRA.

## Setup

```bash
uv sync
```

## Usage

### Training

```bash
# Using main entry point
uv run python main.py train

# Or customize config in scripts/train.py
uv run scripts/train.py
```

### Inference

```bash
uv run python main.py infer "Your text to synthesize"

# With speaker (for multi-speaker models)
uv run python main.py infer "Hello world" "SpeakerName"
```

## Configuration

Edit `src/spark_tts/config.py` or pass values when creating `Config()`:

```python
from src.spark_tts.config import Config

config = Config(
    max_steps=100,
    batch_size=4,
    learning_rate=1e-4,
)
```

## Project Structure

```
src/spark_tts/
├── config.py          # All settings
├── data/
│   ├── tokenizer.py   # Audio tokenization
│   └── dataset.py     # Local dataset loading
├── training/
│   └── trainer.py     # Model loading and training
└── inference/
    └── generate.py    # Speech generation
```
