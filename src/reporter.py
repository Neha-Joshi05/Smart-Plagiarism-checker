"""
reporter.py
Generates detailed plagiarism detection reports.
Saves reports as CSV and text files.
"""

import os
import json
import pandas as pd
from datetime import datetime


def generate_text_report(results, output_path="outputs/report.txt"):
    """Generate a detailed text report."""
    os.makedirs("outputs", exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        "  PLAGIARISM DETECTION REPORT",
        f"  Generated: {now}",
        "=" * 60,
        "",
        "📄 DOCUMENT INFORMATION",
        "-" * 40,
        f"  Original  : {results['doc1']['name']}",
        f"  Submitted : {results['doc2']['name']}",
        f"  Words (Original)  : {results['doc1']['word_count']}",
        f"  Words (Submitted) : {results['doc2']['word_count']}",
        "",
        "📊 PLAGIARISM SCORES",
        "-" * 40,
        f"  ⭐ FINAL SCORE    : {results['final_score']}%",
        f"  ⚖️  VERDICT        : {results['verdict']}",
        f"  🚨 RISK LEVEL     : {results['risk']}",
        "",
        "🔍 ALGORITHM RESULTS",
        "-" * 40,
    ]

    for algo, score in results["algorithm_scores"].items():
        bar = "█" * int(score / 5)
        lines.append(f"  {algo:<15}: {score:6.1f}%  {bar}")

    lines += [
        "",
        "📝 MATCHED PHRASES (KMP)",
        "-" * 40,
    ]

    kmp_matches = results["kmp"]["matches"]
    if kmp_matches:
        for i, match in enumerate(kmp_matches[:10], 1):
            lines.append(
                f"  {i}. \"{match['phrase'][:60]}...\""
                if len(match['phrase']) > 60
                else f"  {i}. \"{match['phrase']}\""
            )
    else:
        lines.append("  No exact phrase matches found.")

    lines += [
        "",
        "🔢 N-GRAM MATCHES (Rabin-Karp)",
        "-" * 40,
        f"  Total n-grams matched: {results['rabin_karp']['match_count']}",
        f"  Total n-grams in doc2: {results['rabin_karp']['total_ngrams']}",
        "",
        "🔑 WINNOWING FINGERPRINTS",
        "-" * 40,
        f"  Doc1 fingerprints: {results['winnowing']['fp1_size']}",
        f"  Doc2 fingerprints: {results['winnowing']['fp2_size']}",
        f"  Common fingerprints: {results['winnowing']['common_fingerprints']}",
        "",
        "=" * 60,
        "  END OF REPORT",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"✅ Text report saved → {output_path}")
    return report_text


def generate_csv_report(results,
    output_path="outputs/report.csv"):
    """Generate CSV report of all scores."""
    os.makedirs("outputs", exist_ok=True)

    rows = []
    for algo, score in results["algorithm_scores"].items():
        rows.append({
            "Algorithm": algo,
            "Score (%)": score,
            "Risk":      (
                "High"   if score >= 70 else
                "Medium" if score >= 40 else
                "Low"    if score >= 15 else
                "None"
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ CSV report saved → {output_path}")
    return df


def generate_json_report(results,
    output_path="outputs/report.json"):
    """Save full results as JSON."""
    os.makedirs("outputs", exist_ok=True)

    # Make results JSON serializable
    clean = {
        "doc1":            results["doc1"],
        "doc2":            results["doc2"],
        "final_score":     results["final_score"],
        "verdict":         results["verdict"],
        "risk":            results["risk"],
        "algorithm_scores":results["algorithm_scores"],
        "kmp_match_count": results["kmp"]["match_count"],
        "rk_match_count":  results["rabin_karp"]["match_count"],
        "win_similarity":  results["winnowing"]["similarity"],
        "generated_at":    datetime.now().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)

    print(f"✅ JSON report saved → {output_path}")
    return clean


def generate_all_reports(results):
    """Generate all report formats."""
    print("\n📄 Generating reports...")
    text = generate_text_report(results)
    df   = generate_csv_report(results)
    data = generate_json_report(results)
    print("✅ All reports generated in outputs/")
    return text, df, data


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.detector import PlagiarismDetector

    detector = PlagiarismDetector()
    text1    = detector.load_document("samples/original.txt")
    text2    = detector.load_document("samples/submitted.txt")
    results  = detector.detect(
        text1, text2, "original.txt", "submitted.txt"
    )
    generate_all_reports(results)