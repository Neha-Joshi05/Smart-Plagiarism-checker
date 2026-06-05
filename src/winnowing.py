"""
winnowing.py
Winnowing Algorithm for Document Fingerprinting.

Used by: Stanford's MOSS plagiarism detection system
Time Complexity: O(N) for fingerprint generation
Space Complexity: O(N)

Generates robust document fingerprints by selecting
minimum hash values from sliding windows.
"""


def get_kgrams(text, k=5):
    """
    Generate k-grams (character-level) from text.
    k=5 means 5-character sequences.
    """
    kgrams = []
    for i in range(len(text) - k + 1):
        kgrams.append(text[i:i+k])
    return kgrams


def hash_kgram(kgram, base=31, prime=1000003):
    """
    Polynomial rolling hash for a k-gram.
    Returns a non-negative integer hash.
    """
    h = 0
    for char in kgram:
        h = (h * base + ord(char)) % prime
    return h


def compute_fingerprints(text, k=5, window_size=4):
    """
    Winnowing algorithm:
    1. Generate k-grams from text
    2. Hash each k-gram
    3. Apply sliding window
    4. Select minimum hash in each window
    5. Deduplicate consecutive same minimums

    Args:
        text        : Input text
        k           : k-gram size
        window_size : Sliding window size

    Returns:
        Set of (hash, position) fingerprints
    """
    from src.preprocessor import clean_text

    text    = clean_text(text)
    kgrams  = get_kgrams(text, k)
    hashes  = [(hash_kgram(kg), i)
               for i, kg in enumerate(kgrams)]

    fingerprints = set()
    prev_min     = None

    # Slide window and pick minimum hash
    for i in range(len(hashes) - window_size + 1):
        window  = hashes[i:i + window_size]
        min_h   = min(window, key=lambda x: x[0])

        if min_h != prev_min:
            fingerprints.add(min_h[0])
            prev_min = min_h

    return fingerprints


def winnowing_similarity(text1, text2,
                         k=5, window_size=4):
    """
    Compute similarity using Winnowing fingerprints.
    Jaccard similarity = |A ∩ B| / |A ∪ B|

    Args:
        text1, text2 : Documents to compare
        k            : k-gram size
        window_size  : Window size for winnowing

    Returns:
        Dict with fingerprints and similarity score
    """
    fp1 = compute_fingerprints(text1, k, window_size)
    fp2 = compute_fingerprints(text2, k, window_size)

    intersection = fp1 & fp2
    union        = fp1 | fp2

    jaccard = (
        len(intersection) / len(union) * 100
        if union else 0.0
    )

    return {
        "algorithm":          "Winnowing",
        "fp1_size":           len(fp1),
        "fp2_size":           len(fp2),
        "common_fingerprints":len(intersection),
        "total_fingerprints": len(union),
        "similarity":         round(jaccard, 2),
        "description":
            "Document fingerprinting (MOSS-style)",
        "k":                  k,
        "window_size":        window_size,
    }


if __name__ == "__main__":
    text1 = """Artificial intelligence is the simulation of
    human intelligence processes by machines."""
    text2 = """Artificial intelligence is the simulation of
    human intelligence by computer systems."""

    result = winnowing_similarity(text1, text2)

    print("=" * 50)
    print("  Winnowing Algorithm Demo")
    print("=" * 50)
    print(f"\n  Fingerprints (Doc1) : {result['fp1_size']}")
    print(f"  Fingerprints (Doc2) : {result['fp2_size']}")
    print(f"  Common fingerprints : {result['common_fingerprints']}")
    print(f"  Similarity          : {result['similarity']}%")