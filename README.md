# 🔍 Plagiarism Detector Using String Matching Algorithms

> **A DSA-based Plagiarism Detection System using KMP, Rabin-Karp, Winnowing, Jaccard, Cosine Similarity, Levenshtein Distance and LCS algorithms with a stunning real-time dashboard!**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![DSA](https://img.shields.io/badge/DSA-String%20Matching-purple)
![Algorithms](https://img.shields.io/badge/Algorithms-7-green)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-orange)

---

## 📌 Project Overview

A complete **Plagiarism Detection System** that includes:
- 🔍 **KMP Algorithm** — Exact phrase matching
- 🔢 **Rabin-Karp** — Rolling hash n-gram matching
- 🔑 **Winnowing (MOSS)** — Document fingerprinting
- 📐 **Jaccard Similarity** — Set-based comparison
- 📊 **Cosine Similarity** — Vector-based comparison
- 📝 **Levenshtein Distance** — Edit distance
- 🔤 **LCS Algorithm** — Longest Common Subsequence
- 📄 **Multi-Document Detection** — Batch comparison
- 📈 **Elite Streamlit Dashboard** — Interactive UI
- 📄 **Auto Report Generation** — TXT, CSV, JSON

**Industry relevance:** Used by Turnitin, iThenticate,
Grammarly, Stanford's MOSS, Unicheck, PlagScan, and
Copyscape for academic and content plagiarism detection.

---

## 🗂️ Folder Structure

```
Plagiarism-Detector/
├── src/
│   ├── preprocessor.py    # Text cleaning & tokenization
│   ├── kmp.py             # KMP string matching
│   ├── rabin_karp.py      # Rabin-Karp rolling hash
│   ├── winnowing.py       # Winnowing fingerprinting
│   ├── similarity.py      # All similarity metrics
│   ├── detector.py        # Main detection engine
│   └── reporter.py        # Report generation
├── samples/
│   ├── original.txt       # Reference document
│   ├── submitted.txt      # Document to check
│   ├── doc1.txt           # Sample document 1
│   ├── doc2.txt           # Sample document 2
│   └── doc3.txt           # Sample document 3
├── outputs/
│   ├── report.txt         # Text report
│   ├── report.csv         # CSV report
│   └── report.json        # JSON report
├── reports/               # Archived reports
├── dashboard.py           # Streamlit dashboard
├── main.py                # CLI interface
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Data | Pandas, NumPy |
| NLP | NLTK |
| ML | Scikit-learn |
| Algorithms | Custom DSA Implementation |

---

## 🧠 Algorithms Implemented

### 1. 🔍 KMP (Knuth-Morris-Pratt)
```
Time Complexity  : O(N + M)
Space Complexity : O(M)
Type             : Exact String Matching
Used For         : Exact phrase detection
```
**How it works:**
- Builds failure function (partial match table)
- No backtracking — uses failure table to skip
- Finds all exact occurrences of pattern in text
- Best for exact phrase plagiarism detection

**Failure Function Example:**
```
Pattern : A B A B C
Table   : 0 0 1 2 0
```

### 2. 🔢 Rabin-Karp
```
Time Complexity  : O(N + M) average, O(NM) worst
Space Complexity : O(1)
Type             : Hash-based Matching
Used For         : N-gram similarity detection
```
**How it works:**
- Polynomial rolling hash for fast matching
- Computes hash for each window of text
- Only verifies character-by-character on hash match
- Excellent for multi-pattern matching

**Rolling Hash Formula:**
```
hash = (char[0]*base^(m-1) + ... + char[m-1]) % prime
```

### 3. 🔑 Winnowing Algorithm
```
Time Complexity  : O(N)
Space Complexity : O(N)
Type             : Document Fingerprinting
Used For         : Robust document similarity
```
**How it works:**
- Generate k-grams from document
- Hash each k-gram using polynomial hash
- Apply sliding window — select minimum hash
- Document fingerprint = set of selected hashes
- Similarity = Jaccard of fingerprint sets

**Used by:** Stanford's MOSS plagiarism system

### 4. 📐 Jaccard Similarity
```
Formula : |A ∩ B| / |A ∪ B| × 100
Range   : 0% to 100%
Type    : Set-based Similarity
Used For: Shingle-level document comparison
```

### 5. 📊 Cosine Similarity
```
Formula : (A · B) / (|A| × |B|) × 100
Range   : 0% to 100%
Type    : Vector-based Similarity
Used For: TF document-level comparison
```

### 6. 📝 Levenshtein Distance
```
Time Complexity  : O(N × M)
Space Complexity : O(N × M)
Type             : Edit Distance
Used For         : Short text / sentence comparison
Operations       : Insert, Delete, Substitute
```

### 7. 🔤 LCS (Longest Common Subsequence)
```
Time Complexity  : O(N × M)
Space Complexity : O(N × M)
Type             : Sequential Matching
Used For         : Finding common sequences
```

---

## 🔄 Detection Workflow

```
Input Documents (Original + Submitted)
              │
              ▼
    Text Preprocessing
    ┌──────────────────┐
    │ • Lowercase      │
    │ • Remove punct   │
    │ • Tokenize       │
    │ • N-grams        │
    │ • Shingles       │
    └──────────────────┘
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
  KMP    Rabin-Karp  Winnowing
(Exact) (N-gram)  (Fingerprint)
    │         │          │
    └─────────┼──────────┘
              │
    Similarity Metrics
    ┌──────────────────┐
    │ • Jaccard        │
    │ • Cosine         │
    │ • Levenshtein    │
    │ • LCS            │
    │ • Word Overlap   │
    └──────────────────┘
              │
              ▼
    Final Score (Weighted Average)
    KMP(30%) + RK(30%) + Win(20%) + Sim(20%)
              │
              ▼
    Verdict + Report Generation
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
(https://github.com/Neha-Joshi05/Smart-Plagiarism-checker.git)
cd Plagiarism-Detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run CLI version
```bash
python main.py
```

### 5. Run dashboard
```bash
python -m streamlit run dashboard.py
```

### 6. Open the app
```
http://localhost:8501
```

---

## 📊 Dashboard Pages

| Page | Features |
|------|---------|
| 🏠 Overview | Algorithm cards, workflow diagram |
| 🔍 Detect Plagiarism | Paste/upload text, gauge chart, radar chart, matched phrases |
| 📊 Algorithm Analysis | Bar chart, donut chart, complexity table |
| 📂 Multi-Document | Batch comparison with progress bars |
| 📄 Reports | View & download TXT, CSV, JSON reports |

---

## 🎯 Plagiarism Score Interpretation

| Score | Risk Level | Verdict |
|-------|-----------|---------|
| 0% - 15% | ✅ None | 🟢 Original Content |
| 15% - 40% | 🟠 Low | 🟠 Low Plagiarism |
| 40% - 70% | 🟡 Medium | 🟡 Moderate Plagiarism |
| 70% - 100% | 🔴 High | 🔴 High Plagiarism |

---

## 📈 Algorithm Comparison

| Algorithm | Time | Space | Type | Accuracy |
|-----------|------|-------|------|---------|
| KMP | O(N+M) | O(M) | Exact Match | ⭐⭐⭐⭐⭐ |
| Rabin-Karp | O(N+M) | O(1) | Hash Match | ⭐⭐⭐⭐⭐ |
| Winnowing | O(N) | O(N) | Fingerprint | ⭐⭐⭐⭐ |
| Jaccard | O(N) | O(N) | Set-based | ⭐⭐⭐⭐ |
| Cosine | O(N) | O(N) | Vector | ⭐⭐⭐⭐ |
| Levenshtein | O(N×M) | O(N×M) | Edit Dist | ⭐⭐⭐ |
| LCS | O(N×M) | O(N×M) | Sequential | ⭐⭐⭐ |

---

## 💡 DSA Concepts Used

```
String Matching
├── KMP Failure Function
├── Rabin-Karp Rolling Hash
├── Exact Pattern Matching
└── Multi-Pattern Matching

Hashing
├── Polynomial Rolling Hash
├── Winnowing Fingerprinting
├── k-gram Hashing
└── Collision Handling

Dynamic Programming
├── Levenshtein Distance (DP table)
├── LCS (DP table)
└── Optimal Substructure

String Processing
├── N-gram Generation
├── Shingling
├── Text Normalization
└── Tokenization

Mathematics
├── Jaccard Coefficient
├── Cosine Similarity
├── TF Vectors
└── Polynomial Hashing
```

---

## 🏢 Real-World Applications

| Platform | Use Case |
|---------|---------|
| Turnitin | Academic paper plagiarism |
| iThenticate | Research paper checking |
| Stanford MOSS | Code plagiarism detection |
| Grammarly | Content originality |
| Unicheck | University submissions |
| Copyscape | Web content plagiarism |
| GitHub Copilot | Code similarity detection |

---

## 📄 Sample Output

```
======================================================
  PLAGIARISM DETECTION REPORT
  Generated: 2024-01-15 14:30:22
======================================================

📄 DOCUMENT INFORMATION
  Original  : original.txt
  Submitted : submitted.txt
  Words (Original)  : 312
  Words (Submitted) : 298

📊 PLAGIARISM SCORES
  ⭐ FINAL SCORE    : 67.43%
  ⚖️  VERDICT        : 🟡 MODERATE PLAGIARISM
  🚨 RISK LEVEL     : Medium

🔍 ALGORITHM RESULTS
  KMP            :  72.4%  ██████████████
  Rabin-Karp     :  68.2%  █████████████
  Winnowing      :  61.5%  ████████████
  Cosine         :  71.3%  ██████████████
  Jaccard        :  58.9%  ███████████
  Levenshtein    :  65.1%  █████████████
  LCS            :  63.7%  ████████████
```

---

## 🎓 Learning Outcomes

- KMP algorithm with failure function
- Rabin-Karp rolling hash technique
- Document fingerprinting (Winnowing)
- Jaccard and Cosine similarity
- Levenshtein edit distance (DP)
- Longest Common Subsequence (DP)
- Text preprocessing pipeline
- N-gram and shingle generation
- Multi-document batch processing
- Interactive Streamlit dashboard

---

## 🏷️ Topics

`python` `dsa` `string-matching` `kmp` `rabin-karp`
`winnowing` `plagiarism-detection` `nlp` `streamlit`
`cosine-similarity` `jaccard` `levenshtein` `lcs`
`text-processing` `algorithms`

---

## 👤 Author

**NEHA JOSHI**
- GitHub: https://github.com/Neha-Joshi05/Smart-Plagiarism-checker.git
- LinkedIn: https://www.linkedin.com/in/neha-joshi-0851a2322?utm_source=share_via&utm_content=profile&utm_medium=member_android

---
