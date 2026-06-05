"""
detector.py
Main plagiarism detection engine.
Orchestrates all algorithms and produces final report.
"""

import os
import sys
sys.path.insert(0, ".")

from src.preprocessor  import preprocess_document
from src.kmp           import find_matching_phrases
from src.rabin_karp    import find_matching_ngrams
from src.winnowing     import winnowing_similarity
from src.similarity    import compute_all_similarities


class PlagiarismDetector:
    """
    Main plagiarism detection engine.
    Combines multiple algorithms for comprehensive detection.
    """

    def __init__(self):
        self.results     = {}
        self.doc1_info   = {}
        self.doc2_info   = {}

    def load_document(self, path_or_text, is_file=True):
        """Load document from file or string."""
        if is_file and os.path.exists(path_or_text):
            with open(path_or_text, "r",
                      encoding="utf-8") as f:
                return f.read()
        elif not is_file:
            return path_or_text
        raise FileNotFoundError(
            f"File not found: {path_or_text}"
        )

    def detect(self, text1, text2,
               doc1_name="Document 1",
               doc2_name="Document 2"):
        """
        Run full plagiarism detection pipeline.

        Args:
            text1     : Original document text
            text2     : Submitted document text
            doc1_name : Name for doc 1
            doc2_name : Name for doc 2

        Returns:
            Comprehensive detection results dict
        """
        print(f"\n{'='*55}")
        print(f"  🔍 Plagiarism Detection Started")
        print(f"{'='*55}")
        print(f"  Doc 1: {doc1_name}")
        print(f"  Doc 2: {doc2_name}")

        # ── 1. Preprocess documents ───────────────────────
        print("\n📝 Preprocessing documents...")
        doc1 = preprocess_document(text1)
        doc2 = preprocess_document(text2)

        self.doc1_info = {
            "name":       doc1_name,
            "word_count": doc1["word_count"],
            "char_count": doc1["char_count"],
            "sent_count": doc1["sent_count"],
            "ngram_count":len(doc1["ngrams"]),
        }
        self.doc2_info = {
            "name":       doc2_name,
            "word_count": doc2["word_count"],
            "char_count": doc2["char_count"],
            "sent_count": doc2["sent_count"],
            "ngram_count":len(doc2["ngrams"]),
        }

        # ── 2. KMP exact phrase matching ──────────────────
        print("🔍 Running KMP Algorithm...")
        kmp_results = find_matching_phrases(
            text1, text2, min_length=5
        )

        # ── 3. Rabin-Karp n-gram matching ─────────────────
        print("🔍 Running Rabin-Karp Algorithm...")
        rk_results = find_matching_ngrams(
            text1, text2, ngram_size=4
        )

        # ── 4. Winnowing fingerprinting ───────────────────
        print("🔍 Running Winnowing Algorithm...")
        win_results = winnowing_similarity(
            text1, text2, k=5, window_size=4
        )

        # ── 5. Similarity metrics ─────────────────────────
        print("📊 Computing similarity metrics...")
        sim_results = compute_all_similarities(text1, text2)

        # ── 6. Calculate final plagiarism score ───────────
        scores = [
            kmp_results["similarity"] * 0.30,
            rk_results["similarity"]  * 0.30,
            win_results["similarity"] * 0.20,
            sim_results["overall"]    * 0.20,
        ]
        final_score = round(sum(scores), 2)

        # ── 7. Determine verdict ──────────────────────────
        if final_score >= 70:
            verdict = "🔴 HIGH PLAGIARISM"
            risk    = "High"
        elif final_score >= 40:
            verdict = "🟡 MODERATE PLAGIARISM"
            risk    = "Medium"
        elif final_score >= 15:
            verdict = "🟠 LOW PLAGIARISM"
            risk    = "Low"
        else:
            verdict = "🟢 ORIGINAL CONTENT"
            risk    = "None"

        print(f"\n{'='*55}")
        print(f"  📊 RESULTS")
        print(f"{'='*55}")
        print(f"  Final Score : {final_score}%")
        print(f"  Verdict     : {verdict}")
        print(f"  KMP         : {kmp_results['similarity']}%")
        print(f"  Rabin-Karp  : {rk_results['similarity']}%")
        print(f"  Winnowing   : {win_results['similarity']}%")
        print(f"  Similarity  : {sim_results['overall']}%")

        self.results = {
            "doc1":         self.doc1_info,
            "doc2":         self.doc2_info,
            "final_score":  final_score,
            "verdict":      verdict,
            "risk":         risk,
            "kmp":          kmp_results,
            "rabin_karp":   rk_results,
            "winnowing":    win_results,
            "similarity":   sim_results,
            "algorithm_scores": {
                "KMP":          kmp_results["similarity"],
                "Rabin-Karp":   rk_results["similarity"],
                "Winnowing":    win_results["similarity"],
                "Cosine":       sim_results["cosine"],
                "Jaccard":      sim_results["jaccard"],
                "Levenshtein":  sim_results["levenshtein"],
                "LCS":          sim_results["lcs"],
                "Word Overlap": sim_results["word_overlap"],
            },
        }

        return self.results

    def detect_multiple(self, original_text,
                        documents):
        """
        Detect plagiarism across multiple documents.

        Args:
            original_text : Reference document
            documents     : List of (name, text) tuples

        Returns:
            List of detection results
        """
        results = []
        for name, text in documents:
            result = self.detect(
                original_text, text,
                "Original", name
            )
            results.append({
                "name":  name,
                "score": result["final_score"],
                "risk":  result["risk"],
                "verdict": result["verdict"],
            })
        return sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )


if __name__ == "__main__":
    # Demo
    detector = PlagiarismDetector()

    text1 = detector.load_document(
        "samples/original.txt"
    )
    text2 = detector.load_document(
        "samples/submitted.txt"
    )

    results = detector.detect(
        text1, text2,
        "original.txt", "submitted.txt"
    )

    print(f"\n✅ Detection complete!")
    print(f"   Final Score : {results['final_score']}%")
    print(f"   Verdict     : {results['verdict']}")