"""
rabin_karp.py
Rabin-Karp Rolling Hash String Matching Algorithm.

Time  Complexity: O(N + M) average, O(NM) worst case
Space Complexity: O(1)

Uses polynomial rolling hash for efficient substring matching.
Excellent for multiple pattern matching simultaneously.
"""

# Prime number for hashing
PRIME = 101
BASE  = 256  # Number of characters in alphabet


def compute_hash(text, length, prime=PRIME, base=BASE):
    """
    Compute polynomial rolling hash for first window.
    hash = (char[0]*base^(m-1) + char[1]*base^(m-2) + ... + char[m-1]) % prime
    """
    h = 0
    for i in range(length):
        h = (h * base + ord(text[i])) % prime
    return h


def rabin_karp_search(text, pattern,
                      prime=PRIME, base=BASE):
    """
    Rabin-Karp rolling hash search.
    Returns list of positions where pattern is found.

    Args:
        text    : Main document text
        pattern : Pattern to search for
        prime   : Prime number for hash modulus
        base    : Base for polynomial hash

    Returns:
        List of starting indices
    """
    if not text or not pattern:
        return []

    n = len(text)
    m = len(pattern)

    if m > n:
        return []

    matches       = []
    pattern_hash  = compute_hash(pattern, m, prime, base)
    text_hash     = compute_hash(text, m, prime, base)

    # Precompute base^(m-1) % prime
    h = pow(base, m-1, prime)

    for i in range(n - m + 1):
        # Check hash match first (quick filter)
        if pattern_hash == text_hash:
            # Verify character by character (avoid hash collision)
            if text[i:i+m] == pattern:
                matches.append(i)

        # Compute rolling hash for next window
        if i < n - m:
            text_hash = (
                base * (text_hash - ord(text[i]) * h) +
                ord(text[i + m])
            ) % prime

            # Handle negative hash
            if text_hash < 0:
                text_hash += prime

    return matches


def find_matching_ngrams(text1, text2,
                         ngram_size=4, min_matches=2):
    """
    Find matching n-grams between two documents
    using Rabin-Karp hashing.

    Args:
        text1      : Original document
        text2      : Submitted document
        ngram_size : Size of n-grams to compare
        min_matches: Minimum matches to report

    Returns:
        Dict with matched n-grams and similarity score
    """
    from src.preprocessor import (
        clean_text, tokenize, get_ngrams
    )

    t1_clean = clean_text(text1)
    t2_clean = clean_text(text2)

    tokens1  = tokenize(t1_clean)
    tokens2  = tokenize(t2_clean)

    ngrams1  = get_ngrams(tokens1, ngram_size)
    ngrams2  = get_ngrams(tokens2, ngram_size)

    # Hash all ngrams from text1
    ngram1_hashes = {}
    for ng in ngrams1:
        h = hash(ng)
        ngram1_hashes[h] = ng

    # Find matching ngrams using rolling hash concept
    matched_ngrams  = []
    matched_count   = 0
    total_matched_words = 0

    for ng2 in ngrams2:
        h = hash(ng2)
        if h in ngram1_hashes:
            # Verify to avoid collision
            if ngram1_hashes[h] == ng2:
                matched_ngrams.append({
                    "ngram":    ng2,
                    "size":     ngram_size,
                    "hash":     h,
                })
                matched_count    += 1
                total_matched_words += ngram_size

    # Calculate similarity
    total_ngrams2 = len(ngrams2) if ngrams2 else 1
    similarity    = min(
        round(matched_count / total_ngrams2 * 100, 2),
        100.0
    )

    return {
        "algorithm":     "Rabin-Karp",
        "ngram_size":    ngram_size,
        "matches":       matched_ngrams[:50],
        "match_count":   matched_count,
        "total_ngrams":  len(ngrams2),
        "similarity":    similarity,
        "description":   "Rolling hash n-gram matching",
    }


def multi_pattern_search(text, patterns):
    """
    Search multiple patterns in text simultaneously.
    Rabin-Karp excels at multi-pattern matching.

    Args:
        text     : Text to search in
        patterns : List of patterns to find

    Returns:
        Dict of pattern → list of positions
    """
    results = {}
    for pattern in patterns:
        positions = rabin_karp_search(text, pattern)
        if positions:
            results[pattern] = positions
    return results


if __name__ == "__main__":
    text    = "machine learning is a subset of artificial intelligence that provides systems"
    pattern = "artificial intelligence"

    positions = rabin_karp_search(text, pattern)

    print("=" * 50)
    print("  Rabin-Karp Algorithm Demo")
    print("=" * 50)
    print(f"\n📝 Text   : {text}")
    print(f"🔍 Pattern: {pattern}")
    print(f"✅ Found at: {positions}")

    # Multi-pattern demo
    patterns  = ["machine learning", "artificial", "systems"]
    results   = multi_pattern_search(text, patterns)
    print(f"\n🔍 Multi-pattern search:")
    for p, pos in results.items():
        print(f"   '{p}' found at: {pos}")