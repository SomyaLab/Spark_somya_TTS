"""
Extract Indic text corpus from transcript JSON files.
Usage: uv run scripts/extract_corpus.py
"""

import json
import re
from pathlib import Path

DATA_DIR = "dataset"
OUTPUT_FILE = "data/indic_corpus.txt"


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    return re.sub(r"\s+", " ", text).strip()


def extract_corpus(data_dir: str = DATA_DIR, output_file: str = OUTPUT_FILE):
    """Extract all transcripts into a single corpus file."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Find all transcript JSON files recursively
    json_files = list(data_path.rglob("*_Transcripts.json"))
    
    if not json_files:
        raise ValueError(f"No *_Transcripts.json files found in {data_dir}")
    
    print(f"Found {len(json_files)} transcript files")
    
    all_texts = set()
    
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        transcripts = data.get("Transcripts", {})
        count = 0
        
        for audio_id, entry in transcripts.items():
            text = entry.get("Transcript", "")
            if text:
                cleaned = clean_text(text)
                if cleaned and len(cleaned) > 5:
                    all_texts.add(cleaned)
                    count += 1
        
        print(f"  {json_file.parent.name}: {count} transcripts")
    
    # Sort and save
    all_texts = sorted(all_texts)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_texts))
    
    print(f"\nExtracted {len(all_texts)} unique sentences to {output_file}")
    
    # Show samples
    print("\nSamples:")
    for text in list(all_texts)[:3]:
        print(f"  {text[:60]}...")


if __name__ == "__main__":
    extract_corpus()
