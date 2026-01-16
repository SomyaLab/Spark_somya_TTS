"""
Expand model embeddings for extended vocabulary.
Usage: uv run scripts/extend_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

BASE_MODEL = "Spark-TTS-0.5B/LLM"
EXTENDED_TOKENIZER = "extended_tokenizer"
OUTPUT = "extended_model"


def extend_model(base: str = BASE_MODEL, tokenizer_path: str = EXTENDED_TOKENIZER, output: str = OUTPUT):
    """Extend model embeddings to match new vocabulary."""
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Not found: {tokenizer_path}\nRun: uv run scripts/extend_tokenizer.py")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    new_vocab = len(tokenizer)
    
    print(f"Loading model from {base}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
    old_vocab = model.config.vocab_size
    
    if new_vocab <= old_vocab:
        print("No expansion needed")
        return
    
    print(f"Expanding: {old_vocab} -> {new_vocab} tokens")
    model.resize_token_embeddings(new_vocab)
    
    # Initialize new embeddings with mean + noise
    with torch.no_grad():
        embed = model.get_input_embeddings()
        mean_emb = embed.weight[:old_vocab].mean(dim=0)
        for i in range(old_vocab, new_vocab):
            embed.weight[i] = mean_emb + torch.randn_like(mean_emb) * 0.01
    
    model.config.vocab_size = new_vocab
    
    Path(output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Saved to {output}")
    
    # Verify
    print("\nVerifying...")
    model = AutoModelForCausalLM.from_pretrained(output)
    for text in ["महाभारत एक विराट", "ಕನ್ನಡ ಭಾಷೆ"]:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        print(f"  '{text}' -> {out.logits.shape}")


if __name__ == "__main__":
    extend_model()
