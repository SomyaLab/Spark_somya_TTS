"""
Extend Qwen tokenizer with Indic vocabulary.
Usage: uv run scripts/extend_tokenizer.py
"""

import sentencepiece as spm
from transformers import AutoTokenizer
from pathlib import Path

BASE_TOKENIZER = "Spark-TTS-0.5B/LLM"
INDIC_MODEL = "data/indic_tokenizer.model"
OUTPUT = "extended_tokenizer"


def extend_tokenizer(base_path: str = BASE_TOKENIZER, indic_path: str = INDIC_MODEL, output: str = OUTPUT):
    """Extend base tokenizer with Indic vocabulary."""
    if not Path(indic_path).exists():
        raise FileNotFoundError(f"Not found: {indic_path}\nRun: uv run scripts/train_tokenizer.py")
    
    # Load base tokenizer
    print(f"Loading base tokenizer from {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    base_vocab = set(tokenizer.get_vocab().keys())
    print(f"Base vocab: {len(base_vocab)}")
    
    # Load Indic vocab (skip special tokens)
    sp = spm.SentencePieceProcessor(model_file=indic_path)
    indic_vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size()) 
                   if not (sp.id_to_piece(i).startswith("<") and sp.id_to_piece(i).endswith(">"))]
    
    # Add new tokens
    new_tokens = [t for t in indic_vocab if t not in base_vocab]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} new tokens -> {len(tokenizer)} total")
    
    # Save
    Path(output).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    print(f"Saved to {output}")
    
    # Compare efficiency
    base_tok = AutoTokenizer.from_pretrained(base_path)
    print("\nEfficiency comparison:")
    for text in ["महाभारत एक विराट कृति है", "ಕನ್ನಡ ಭಾಷೆ ಸುಂದರವಾಗಿದೆ"]:
        before = len(base_tok.tokenize(text))
        after = len(tokenizer.tokenize(text))
        print(f"  '{text[:15]}...' : {before} -> {after} tokens ({before/after:.1f}x)")
    
    return len(tokenizer)


if __name__ == "__main__":
    extend_tokenizer()
