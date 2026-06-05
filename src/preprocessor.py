"""
preprocessor.py
Text preprocessing pipeline for plagiarism detection.
Cleans, tokenizes and normalizes text documents.
"""

import re
import string
from collections import Counter


def clean_text(text):
    """
    Clean and normalize text:
    - Convert to lowercase
    - Remove punctuation
    - Remove extra whitespace
    - Remove special characters
    """
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def tokenize(text):
    """Split text into list of words."""
    return text.split()


def get_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def get_ngrams(tokens, n=3):
    """
    Generate n-grams from token list.
    n=3 means trigrams (3-word phrases)
    Used for phrase-level matching.
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams


def get_shingles(text, k=5):
    """
    Generate k-shingles (k-word sequences).
    Used for document-level similarity.
    Returns a set of unique shingles.
    """
    tokens  = tokenize(clean_text(text))
    shingles = set()
    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i:i+k])
        shingles.add(shingle)
    return shingles


def word_frequency(text):
    """Return word frequency counter."""
    tokens = tokenize(clean_text(text))
    return Counter(tokens)


def preprocess_document(text):
    """
    Full preprocessing pipeline.
    Returns dict with all processed forms.
    """
    cleaned   = clean_text(text)
    tokens    = tokenize(cleaned)
    sentences = get_sentences(text)
    ngrams    = get_ngrams(tokens, n=3)
    shingles  = get_shingles(text, k=5)
    freq      = word_frequency(text)

    return {
        "original":  text,
        "cleaned":   cleaned,
        "tokens":    tokens,
        "sentences": sentences,
        "ngrams":    ngrams,
        "shingles":  shingles,
        "frequency": freq,
        "word_count":len(tokens),
        "char_count":len(text),
        "sent_count":len(sentences),
    }


if __name__ == "__main__":
    sample = """
    Artificial intelligence is the simulation of human intelligence.
    Machine learning is a subset of artificial intelligence.
    Deep learning is part of machine learning methods.
    """
    result = preprocess_document(sample)
    print("✅ Preprocessing complete!")
    print(f"   Words     : {result['word_count']}")
    print(f"   Sentences : {result['sent_count']}")
    print(f"   N-grams   : {len(result['ngrams'])}")
    print(f"   Shingles  : {len(result['shingles'])}")
    print(f"\n📝 Tokens (first 10):")
    print(result['tokens'][:10])
    print(f"\n📝 N-grams (first 5):")
    print(result['ngrams'][:5])