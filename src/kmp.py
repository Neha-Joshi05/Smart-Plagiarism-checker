"""
kmp.py
Knuth-Morris-Pratt (KMP) String Matching Algorithm.

Time  Complexity: O(N + M)
Space Complexity: O(M)

Where:
  N = length of text
  M = length of pattern

Used for: Exact phrase matching in documents.
"""


def build_failure_function(pattern):
    """
    Build the KMP failure function (partial match table).
    This table tells us how many characters we can skip
    when a mismatch occurs.

    Example:
    Pattern: "ABABC"
    Table:   [0, 0, 1, 2, 0]
    """
    m       = len(pattern)
    failure = [0] * m
    j       = 0
    i       = 1

    while i < m:
        if pattern[i] == pattern[j]:
            j += 1
            failure[i] = j
            i += 1
        else:
            if j != 0:
                j = failure[j - 1]
            else:
                failure[i] = 0
                i += 1

    return failure


def kmp_search(text, pattern):
    """
    KMP pattern search algorithm.
    Returns list of starting indices where pattern is found.

    Args:
        text    : The main text to search in
        pattern : The pattern/phrase to find

    Returns:
        List of indices where pattern starts in text
    """
    if not pattern or not text:
        return []

    n       = len(text)
    m       = len(pattern)
    failure = build_failure_function(pattern)
    matches = []

    i = 0  # index for text
    j = 0  # index for pattern

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        else:
            if j != 0:
                j = failure[j - 1]
            else:
                i += 1

        if j == m:
            matches.append(i - j)
            j = failure[j - 1]

    return matches


def find_matching_phrases(text1, text2, min_length=5):
    """
    Find all common phrases between two documents using KMP.
    Splits text2 into sentences and searches each in text1.

    Args:
        text1      : Original document
        text2      : Submitted document
        min_length : Minimum phrase length to consider

    Returns:
        List of dicts with matched phrase info
    """
    from src.preprocessor import clean_text, get_sentences

    t1_clean   = clean_text(text1)
    t2_clean   = clean_text(text2)
    sentences2 = get_sentences(text2)

    matches = []
    total_matched_chars = 0

    for sentence in sentences2:
        cleaned_sent = clean_text(sentence)
        words = cleaned_sent.split()

        if len(words) < min_length:
            continue

        # Search phrase in text1 using KMP
        occurrences = kmp_search(t1_clean, cleaned_sent)

        if occurrences:
            matches.append({
                "phrase":      sentence.strip(),
                "length":      len(words),
                "positions":   occurrences,
                "match_count": len(occurrences),
            })
            total_matched_chars += len(cleaned_sent)

    # Calculate similarity percentage
    total_chars = len(t2_clean) if t2_clean else 1
    similarity  = min(
        round(total_matched_chars / total_chars * 100, 2),
        100.0
    )

    return {
        "algorithm":   "KMP",
        "matches":     matches,
        "match_count": len(matches),
        "similarity":  similarity,
        "description": "Knuth-Morris-Pratt exact phrase matching",
    }


if __name__ == "__main__":
    text    = "artificial intelligence is the simulation of human intelligence processes"
    pattern = "intelligence processes"

    failure  = build_failure_function(pattern)
    positions = kmp_search(text, pattern)

    print("=" * 50)
    print("  KMP Algorithm Demo")
    print("=" * 50)
    print(f"\n📝 Text   : {text}")
    print(f"🔍 Pattern: {pattern}")
    print(f"\n📊 Failure Function: {failure}")
    print(f"✅ Found at position(s): {positions}")
    if positions:
        for pos in positions:
            print(f"   Match: '{text[pos:pos+len(pattern)]}'")