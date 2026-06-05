"""
main.py
CLI interface for the Plagiarism Detector.
Run: python main.py
"""

import sys
import os
sys.path.insert(0, ".")

from src.detector  import PlagiarismDetector
from src.reporter  import generate_all_reports


def main():
    print("=" * 55)
    print("  🔍 Plagiarism Detector — DSA Project")
    print("=" * 55)

    detector = PlagiarismDetector()

    # ── Load sample documents ─────────────────────────────
    print("\n📂 Loading documents...")
    text1 = detector.load_document("samples/original.txt")
    text2 = detector.load_document("samples/submitted.txt")

    # ── Run detection ─────────────────────────────────────
    results = detector.detect(
        text1, text2,
        "original.txt",
        "submitted.txt"
    )

    # ── Generate reports ──────────────────────────────────
    generate_all_reports(results)

    # ── Multi-document detection ──────────────────────────
    print("\n\n📂 Multi-Document Detection...")
    print("-" * 45)

    docs = []
    for fname in ["doc1.txt","doc2.txt","doc3.txt"]:
        path = f"samples/{fname}"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                docs.append((fname, f.read()))

    if len(docs) >= 2:
        multi = detector.detect_multiple(
            text1, docs
        )
        print("\n📊 Multi-Document Results:")
        print(f"{'Document':<15} {'Score':>8} {'Risk':>10} {'Verdict'}")
        print("-" * 55)
        for r in multi:
            print(
                f"{r['name']:<15} "
                f"{r['score']:>7.1f}% "
                f"{r['risk']:>10}  "
                f"{r['verdict']}"
            )

    print("\n✅ All done!")
    print("   📁 Check outputs/ for reports")
    print("   🚀 Run dashboard: python -m streamlit run dashboard.py")


if __name__ == "__main__":
    main()