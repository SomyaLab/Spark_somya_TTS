"""
Text Processing (Legacy Location)

This module has been moved to src/spark_tts/text_processor.py
This file re-exports for backward compatibility.
"""

# Re-export from new location
from spark_tts.text_processor import (
    IndicLanguageDetector,
    TextNormalizer,
    Transliterator,
    LongTextProcessor,
    get_transliterator,
    get_long_text_processor,
    normalize_text,
    detect_language,
)

__all__ = [
    "IndicLanguageDetector",
    "TextNormalizer", 
    "Transliterator",
    "LongTextProcessor",
    "get_transliterator",
    "get_long_text_processor",
    "normalize_text",
    "detect_language",
]


if __name__ == "__main__":
    # Quick test
    ts = Transliterator(default_lang="kn")
    
    print(f"{'='*20} TESTING {'='*20}\n")

    txt1 = "ISRO launch madida rocket 100% success aagide."
    print(f"Input:  {txt1}")
    res1 = ts.process_text(txt1, lang="kn", transliterate_english=True)
    print(f"Output: {res1}\n")

    txt2 = "Iska price ₹500 hai."
    print(f"Input:  {txt2}")
    res2 = ts.process_text(txt2, lang="hi", transliterate_english=True)
    print(f"Output: {res2}\n")