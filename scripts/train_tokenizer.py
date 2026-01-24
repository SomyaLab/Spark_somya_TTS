"""
Train SentencePiece BPE tokenizer on Indic corpus.
Usage: uv run scripts/train_tokenizer.py
"""

import sentencepiece as spm
from pathlib import Path

CORPUS_FILE = "data/indic_corpus.txt"
MODEL_PREFIX = "data/indic_tokenizer"
VOCAB_SIZE = 10000


def train_tokenizer(corpus_file: str = CORPUS_FILE, model_prefix: str = MODEL_PREFIX):
    """Train SentencePiece tokenizer on corpus."""
    if not Path(corpus_file).exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_file}\nRun: uv run scripts/extract_corpus.py")
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    print(f"Training on {num_lines} sentences")
    
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        character_coverage=0.9995,
        model_type="bpe",
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        byte_fallback=True,
        normalization_rule_name="identity",
    )
    
    print(f"Saved: {model_prefix}.model")
    
    # Test
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    for text in ["महाभारत एक विराट कृति है", "ಕನ್ನಡ ಭಾಷೆ ಸುಂದರವಾಗಿದೆ"]:
        tokens = sp.encode_as_pieces(text)
        print(f"  '{text[:20]}...' -> {len(tokens)} tokens")


if __name__ == "__main__":
    train_tokenizer()
