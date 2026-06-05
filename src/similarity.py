"""
similarity.py
Multiple similarity calculation methods:
1. Jaccard Similarity (Set-based)
2. Cosine Similarity (Vector-based)
3. Levenshtein Distance (Edit distance)
4. Longest Common Subsequence (LCS)
"""

import math
from collections import Counter


def jaccard_similarity(set1, set2):
    """
    Jaccard Similarity = |A ∩ B| / |A ∪ B|
    Range: 0.0 to 1.0
    Used for: Set/shingle-based comparison
    """
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union        = len(set1 | set2)
    return round(intersection / union * 100, 2) if union else 0.0


def cosine_similarity(text1, text2):
    """
    Cosine Similarity using TF vectors.
    sim = (A · B) / (|A| × |B|)
    Range: 0.0 to 1.0
    Used for: Document-level similarity
    """
    from src.preprocessor import clean_text, tokenize

    tokens1 = tokenize(clean_text(text1))
    tokens2 = tokenize(clean_text(text2))

    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)

    all_words  = set(freq1.keys()) | set(freq2.keys())
    vec1       = [freq1.get(w, 0) for w in all_words]
    vec2       = [freq2.get(w, 0) for w in all_words]

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1        = math.sqrt(sum(a**2 for a in vec1))
    mag2        = math.sqrt(sum(b**2 for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return round(dot_product / (mag1 * mag2) * 100, 2)


def levenshtein_distance(s1, s2, max_len=500):
    """
    Levenshtein Edit Distance.
    Minimum edit operations to transform s1 into s2.
    Operations: insert, delete, substitute

    Returns similarity as percentage.
    """
    # Limit length for performance
    s1 = s1[:max_len]
    s2 = s2[:max_len]

    m, n = len(s1), len(s2)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1],  # substitute
                )

    distance    = dp[m][n]
    max_dist    = max(m, n) if max(m, n) > 0 else 1
    similarity  = round(
        (1 - distance / max_dist) * 100, 2
    )
    return similarity, distance


def lcs_length(s1, s2, max_len=300):
    """
    Longest Common Subsequence length.
    Returns LCS length and similarity percentage.
    """
    s1 = s1[:max_len]
    s2 = s2[:max_len]

    m, n = len(s1), len(s2)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len    = dp[m][n]
    similarity = round(
        lcs_len / max(m, n) * 100, 2
    ) if max(m, n) > 0 else 0.0

    return lcs_len, similarity


def word_overlap_similarity(text1, text2):
    """
    Simple word overlap similarity.
    Common words / total unique words.
    """
    from src.preprocessor import clean_text, tokenize

    words1 = set(tokenize(clean_text(text1)))
    words2 = set(tokenize(clean_text(text2)))

    common = words1 & words2
    total  = words1 | words2

    return round(
        len(common) / len(total) * 100, 2
    ) if total else 0.0


def compute_all_similarities(text1, text2):
    """
    Run all similarity algorithms and return results.
    """
    from src.preprocessor import get_shingles, clean_text

    shingles1 = get_shingles(text1, k=4)
    shingles2 = get_shingles(text2, k=4)
    jaccard   = jaccard_similarity(shingles1, shingles2)

    cosine    = cosine_similarity(text1, text2)

    c1        = clean_text(text1)[:300]
    c2        = clean_text(text2)[:300]
    lev_sim, lev_dist = levenshtein_distance(c1, c2)

    lcs_len, lcs_sim  = lcs_length(c1, c2)

    word_sim  = word_overlap_similarity(text1, text2)

    # Weighted average
    overall   = round(
        jaccard * 0.25 + cosine * 0.30 +
        lev_sim * 0.20 + lcs_sim * 0.15 +
        word_sim * 0.10,
        2
    )

    return {
        "jaccard":          jaccard,
        "cosine":           cosine,
        "levenshtein":      lev_sim,
        "lcs":              lcs_sim,
        "word_overlap":     word_sim,
        "overall":          overall,
        "lev_distance":     lev_dist,
        "lcs_length":       lcs_len,
    }


if __name__ == "__main__":
    text1 = "machine learning is a subset of artificial intelligence"
    text2 = "machine learning is an area of artificial intelligence"

    results = compute_all_similarities(text1, text2)
    print("=" * 50)
    print("  Similarity Metrics Demo")
    print("=" * 50)
    for key, val in results.items():
        print(f"  {key:<20}: {val}")