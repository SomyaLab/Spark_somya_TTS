"""
Upload Spark-Somya-TTS to HuggingFace Hub.

Bundles the finetuned LLM model with BiCodec components for easy deployment.
"""

import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, create_repo


# Default configuration
HF_REPO_ID = os.environ.get("HF_REPO_ID", "somyalab/Spark_somya_TTS")
BICODEC_SOURCE = "SparkAudio/Spark-TTS-0.5B"
FINETUNED_MODEL_DIR = "finetuned_model"

# Files to copy from finetuned_model
LLM_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]

# BiCodec components to copy
BICODEC_COMPONENTS = [
    "config.yaml",
    "BiCodec",
    "wav2vec2-large-xlsr-53",
]


def download_bicodec(cache_dir: str) -> str:
    """Download BiCodec from SparkAudio/Spark-TTS-0.5B."""
    print(f"Downloading BiCodec from {BICODEC_SOURCE}...")
    
    # Only download the files we need
    bicodec_path = snapshot_download(
        repo_id=BICODEC_SOURCE,
        allow_patterns=[
            "config.yaml",
            "BiCodec/**",
            "wav2vec2-large-xlsr-53/**",
        ],
        cache_dir=cache_dir,
    )
    
    print(f"BiCodec downloaded to: {bicodec_path}")
    return bicodec_path


def prepare_upload_dir(
    finetuned_model_dir: str,
    bicodec_dir: str,
    output_dir: str,
) -> None:
    """Prepare the directory structure for upload."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    finetuned_path = Path(finetuned_model_dir)
    bicodec_path = Path(bicodec_dir)
    
    # Copy LLM files from finetuned_model
    print("Copying finetuned LLM files...")
    for filename in LLM_FILES:
        src = finetuned_path / filename
        if src.exists():
            dst = output_path / filename
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")
        else:
            print(f"  Warning: {filename} not found in finetuned_model")
    
    # Copy BiCodec components
    print("Copying BiCodec components...")
    for component in BICODEC_COMPONENTS:
        src = bicodec_path / component
        dst = output_path / component
        
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied directory: {component}")
        elif src.is_file():
            shutil.copy2(src, dst)
            print(f"  Copied file: {component}")
        else:
            print(f"  Warning: {component} not found in BiCodec")


def create_model_card(output_dir: str) -> None:
    """Create README.md (model card) for HuggingFace."""
    
    model_card = '''---
license: apache-2.0
language:
  - hi
  - kn
  - ta
  - bn
  - gu
  - te
  - mr
  - en
tags:
  - text-to-speech
  - tts
  - indic
  - zero-shot
  - voice-cloning
pipeline_tag: text-to-speech
---

# Spark-Somya-TTS

Zero-shot voice cloning TTS model for Indic languages, fine-tuned from Spark-TTS-0.5B.

## Supported Languages

- Hindi (hi)
- Kannada (kn)
- Tamil (ta)
- Bengali (bn)
- Gujarati (gu)
- Telugu (te)
- Marathi (mr)
- English (en)

## Quick Start

### Installation

```bash
pip install torch transformers huggingface_hub unsloth soundfile librosa numpy
```

### Download Model

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download("somyalab/Spark_somya_TTS")
```

### Inference

```python
import torch
import numpy as np
import soundfile as sf
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

# Load audio tokenizer (BiCodec)
import sys
sys.path.insert(0, model_dir)
from sparktts.models.audio_tokenizer import BiCodecTokenizer

audio_tokenizer = BiCodecTokenizer(model_dir, "cuda")

# Reference audio for voice cloning
import librosa
ref_audio, ref_sr = librosa.load("reference_voice.wav", sr=None)
ref_global_tokens, _ = audio_tokenizer.tokenize_audio(ref_audio, ref_sr)

# Generate speech
text = "नमस्ते, यह एक परीक्षण है।"

prompt = "".join([
    "<|task_tts|>",
    "<|start_content|>",
    text,
    "<|end_content|>",
    "<|start_global_token|>",
    ref_global_tokens,
    "<|end_global_token|>",
    "<|start_semantic_token|>",
])

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
)

# Decode to audio
generated_ids = outputs[:, inputs.input_ids.shape[1]:]
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0].tolist())

# Extract semantic token IDs
semantic_ids = []
for t in generated_tokens:
    if t.startswith("<|bicodec_semantic_") and t.endswith("|>"):
        semantic_ids.append(int(t[18:-2]))

# Detokenize to waveform
import re
global_matches = re.findall(r"<\\|bicodec_global_(\\d+)\\|>", ref_global_tokens)
global_ids = torch.tensor([int(t) for t in global_matches]).unsqueeze(0).unsqueeze(0)
semantic_ids = torch.tensor(semantic_ids).unsqueeze(0)

wav = audio_tokenizer.detokenize(
    global_ids.to("cuda").squeeze(0),
    semantic_ids.to("cuda"),
)

sf.write("output.wav", wav, 16000)
```

## Model Architecture

- Base: Qwen2ForCausalLM (0.5B parameters)
- Fine-tuned for Indic languages with extended tokenizer
- Uses BiCodec for audio tokenization/detokenization

## Citation

If you use this model, please cite:

```bibtex
@misc{spark-somya-tts,
  title={Spark-Somya-TTS},
  author={Somya Lab},
  year={2025},
  url={https://huggingface.co/somyalab/Spark_somya_TTS}
}
```

## License

Apache 2.0
'''
    
    readme_path = Path(output_dir) / "README.md"
    readme_path.write_text(model_card)
    print(f"Created model card: {readme_path}")


def upload_to_hub(
    upload_dir: str,
    repo_id: str,
    private: bool = False,
) -> str:
    """Upload the prepared directory to HuggingFace Hub."""
    
    api = HfApi()
    
    print(f"Creating/updating repository: {repo_id}")
    
    # Create repo if it doesn't exist
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    
    # Upload all files
    print(f"Uploading files from {upload_dir}...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Spark-Somya-TTS model",
    )
    
    print(f"\nUpload complete!")
    print(f"Model URL: https://huggingface.co/{repo_id}")
    
    return repo_id


def main():
    """Main upload workflow."""
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    finetuned_model_dir = project_root / FINETUNED_MODEL_DIR
    
    if not finetuned_model_dir.exists():
        raise FileNotFoundError(
            f"Finetuned model not found at {finetuned_model_dir}. "
            "Please ensure you have trained the model first."
        )
    
    # Create temp directory for staging
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        upload_dir = Path(tmpdir) / "upload"
        
        # Download BiCodec
        bicodec_dir = download_bicodec(str(cache_dir))
        
        # Prepare upload directory
        prepare_upload_dir(
            finetuned_model_dir=str(finetuned_model_dir),
            bicodec_dir=bicodec_dir,
            output_dir=str(upload_dir),
        )
        
        # Create model card
        create_model_card(str(upload_dir))
        
        # Upload to HuggingFace
        repo_id = os.environ.get("HF_REPO_ID", HF_REPO_ID)
        private = os.environ.get("HF_PRIVATE", "false").lower() == "true"
        
        repo_id = upload_to_hub(
            upload_dir=str(upload_dir),
            repo_id=repo_id,
            private=private,
        )
        
        print(f"\nDone! Your model is available at:")
        print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
