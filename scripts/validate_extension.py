"""
Validate tokenizer extension pipeline.
Usage: uv run scripts/validate_extension.py
"""

from pathlib import Path


def check_step(name: str, path: str, check_fn) -> bool:
    """Check a pipeline step."""
    print(f"\n{name}")
    print("-" * 40)
    if not Path(path).exists():
        print(f"[ ] Not found: {path}")
        return False
    try:
        return check_fn(path)
    except Exception as e:
        print(f"[!] Error: {e}")
        return False


def main():
    print("=" * 50)
    print("Indic Tokenizer Extension Validation")
    print("=" * 50)
    
    # Step 1: Corpus
    def check_corpus(p):
        lines = open(p).readlines()
        print(f"[x] Corpus: {len(lines)} lines")
        return len(lines) > 0
    
    # Step 2: SentencePiece
    def check_sp(p):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=p)
        print(f"[x] SentencePiece: {sp.get_piece_size()} tokens")
        return True
    
    # Step 3: Extended tokenizer
    def check_tokenizer(p):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(p)
        base = AutoTokenizer.from_pretrained("Spark-TTS-0.5B/LLM")
        print(f"[x] Tokenizer: {len(tok)} tokens")
        for text in ["महाभारत एक विराट", "ಕನ್ನಡ ಭಾಷೆ"]:
            b, e = len(base.tokenize(text)), len(tok.tokenize(text))
            print(f"    '{text[:10]}...' : {b} -> {e} ({b/e:.1f}x)")
        return True
    
    # Step 4: Extended model
    def check_model(p):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForCausalLM.from_pretrained(p)
        assert len(tok) == model.config.vocab_size
        print(f"[x] Model: vocab={model.config.vocab_size}")
        inputs = tok("test", return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
        print("    Forward pass OK")
        return True
    
    steps = [
        ("Step 1: Corpus", "data/indic_corpus.txt", check_corpus),
        ("Step 2: SentencePiece", "data/indic_tokenizer.model", check_sp),
        ("Step 3: Tokenizer", "extended_tokenizer", check_tokenizer),
        ("Step 4: Model", "extended_model", check_model),
    ]
    
    for name, path, fn in steps:
        if not check_step(name, path, fn):
            print(f"\n[!] Run the missing step first")
            return
    
    print("\n" + "=" * 50)
    print("All checks passed! Run: python main.py train-extended")


if __name__ == "__main__":
    main()
