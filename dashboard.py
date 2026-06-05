"""
dashboard.py — Plagiarism Detector Elite Dashboard
Run: python -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import time

sys.path.insert(0, ".")
from src.detector  import PlagiarismDetector
from src.reporter  import generate_all_reports
from src.preprocessor import preprocess_document

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Elite CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
  * { font-family: 'Inter', sans-serif !important; }
  .stApp { background: #020408; }

  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg,#0a0f1e,#050810) !important;
      border-right: 1px solid #1a2744;
  }
  [data-testid="stMetric"] {
      background: linear-gradient(135deg,#0a0f1e,#0d1526);
      border: 1px solid #1a2744;
      border-radius: 16px;
      padding: 20px !important;
      position: relative; overflow: hidden;
  }
  [data-testid="stMetric"]::before {
      content:''; position:absolute;
      top:0; left:0; right:0; height:2px;
      background: linear-gradient(90deg,#8b5cf6,#ec4899,#8b5cf6);
  }
  [data-testid="stMetricValue"] {
      font-size:2rem !important; font-weight:900 !important;
      background: linear-gradient(135deg,#8b5cf6,#ec4899);
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;
  }
  [data-testid="stMetricLabel"] {
      color:#8892b0 !important;
      font-size:0.75rem !important;
      text-transform:uppercase; letter-spacing:1px;
  }
  .hero-card {
      background: linear-gradient(135deg,#0a0f1e,#0d1526,#0a0f1e);
      border: 1px solid #1a2744; border-radius:20px;
      padding:30px; margin:10px 0;
      position:relative; overflow:hidden;
  }
  .hero-card::before {
      content:''; position:absolute;
      top:0; left:0; right:0; height:3px;
      background: linear-gradient(90deg,#8b5cf6,#ec4899,#f59e0b,#8b5cf6);
      background-size:200% 100%;
      animation: move 3s linear infinite;
  }
  @keyframes move {
      0%{background-position:0%} 100%{background-position:200%}
  }
  .verdict-high {
      background:linear-gradient(135deg,#450a0a,#7f1d1d);
      border:2px solid #ef4444; border-radius:16px;
      padding:20px; text-align:center;
  }
  .verdict-medium {
      background:linear-gradient(135deg,#451a03,#78350f);
      border:2px solid #f59e0b; border-radius:16px;
      padding:20px; text-align:center;
  }
  .verdict-low {
      background:linear-gradient(135deg,#1c1917,#292524);
      border:2px solid #f97316; border-radius:16px;
      padding:20px; text-align:center;
  }
  .verdict-none {
      background:linear-gradient(135deg,#052e16,#14532d);
      border:2px solid #22c55e; border-radius:16px;
      padding:20px; text-align:center;
  }
  .match-card {
      background:#0a0f1e; border:1px solid #1a2744;
      border-left:4px solid #8b5cf6;
      border-radius:10px; padding:12px 16px; margin:6px 0;
  }
  .algo-card {
      background:#0a0f1e; border:1px solid #1a2744;
      border-radius:12px; padding:16px; margin:6px 0;
  }
  .section-title {
      font-size:1.1rem; font-weight:700; color:#ccd6f6;
      margin:16px 0 8px; padding-left:10px;
      border-left:3px solid #8b5cf6;
  }
  .stButton>button {
      background:linear-gradient(135deg,#8b5cf6,#ec4899) !important;
      color:white !important; border:none !important;
      border-radius:12px !important; font-weight:700 !important;
      padding:12px 28px !important; width:100% !important;
  }
  .stTextArea>div>textarea {
      background:#0a0f1e !important;
      border:1px solid #1a2744 !important;
      color:#ccd6f6 !important;
      border-radius:12px !important;
  }
  div[data-testid="stDataFrame"] {
      border:1px solid #1a2744 !important;
      border-radius:12px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,30,0.8)",
    font=dict(family="Inter", color="#8892b0"),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center;padding:20px 0'>
  <div style='font-size:3rem'>🔍</div>
  <div style='font-size:1.3rem;font-weight:900;
    background:linear-gradient(135deg,#8b5cf6,#ec4899);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent'>
    PlagDetect
  </div>
  <div style='color:#8892b0;font-size:0.8rem'>
    DSA String Matching Engine
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "🏠 Overview",
    "🔍 Detect Plagiarism",
    "📊 Algorithm Analysis",
    "📂 Multi-Document",
    "📄 Reports",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='color:#8892b0;font-size:0.75rem'>
  <b style='color:#8b5cf6'>Algorithms:</b><br>
  • KMP String Matching<br>
  • Rabin-Karp Hashing<br>
  • Winnowing (MOSS)<br>
  • Jaccard Similarity<br>
  • Cosine Similarity<br>
  • Levenshtein Distance<br>
  • LCS Algorithm<br><br>
  <b style='color:#8b5cf6'>Complexity:</b><br>
  • KMP: O(N + M)<br>
  • Rabin-Karp: O(N + M)<br>
  • Winnowing: O(N)<br>
  • LCS: O(N × M)
</div>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def get_verdict_class(score):
    if score >= 70: return "verdict-high"
    if score >= 40: return "verdict-medium"
    if score >= 15: return "verdict-low"
    return "verdict-none"

def get_verdict_color(score):
    if score >= 70: return "#ef4444"
    if score >= 40: return "#f59e0b"
    if score >= 15: return "#f97316"
    return "#22c55e"

def get_verdict_text(score):
    if score >= 70: return "🔴 HIGH PLAGIARISM"
    if score >= 40: return "🟡 MODERATE PLAGIARISM"
    if score >= 15: return "🟠 LOW PLAGIARISM"
    return "🟢 ORIGINAL CONTENT"

def load_sample(filename):
    path = f"samples/{filename}"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero-card'>
      <h1 style='margin:0;font-size:2.2rem;font-weight:900;
        background:linear-gradient(135deg,#8b5cf6,#ec4899,#f59e0b);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent'>
        🔍 Plagiarism Detector
      </h1>
      <p style='color:#8892b0;margin:8px 0 0;font-size:1rem'>
        KMP • Rabin-Karp • Winnowing • Jaccard • Cosine • LCS
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🧠 Algorithms",   7)
    c2.metric("📝 Text Methods", 4)
    c3.metric("🔢 Hash Methods", 2)
    c4.metric("📊 Metrics",      5)
    c5.metric("⚡ Speed",        "O(N+M)")

    st.markdown("---")

    # Algorithm cards
    st.markdown('<div class="section-title">🧠 Algorithms Used</div>',
                unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        for name, complexity, desc, color in [
            ("KMP Algorithm",      "O(N + M)",
             "Exact phrase matching using failure function. No backtracking needed.",
             "#8b5cf6"),
            ("Rabin-Karp",         "O(N + M) avg",
             "Rolling hash for n-gram matching. Perfect for multi-pattern search.",
             "#ec4899"),
            ("Winnowing (MOSS)",   "O(N)",
             "Document fingerprinting used by Stanford's MOSS system.",
             "#f59e0b"),
            ("Jaccard Similarity", "O(N)",
             "Set-based similarity using shingles intersection over union.",
             "#10b981"),
        ]:
            st.markdown(f"""
            <div class='algo-card'>
              <div style='display:flex;justify-content:space-between;
                align-items:center;margin-bottom:6px'>
                <span style='color:{color};font-weight:700'>
                  {name}
                </span>
                <span style='background:#1a2744;color:#8892b0;
                  padding:2px 8px;border-radius:10px;
                  font-size:0.75rem;font-family:monospace'>
                  {complexity}
                </span>
              </div>
              <span style='color:#8892b0;font-size:0.85rem'>
                {desc}
              </span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        for name, complexity, desc, color in [
            ("Cosine Similarity",  "O(N)",
             "Vector-based TF similarity between documents.",
             "#3b82f6"),
            ("Levenshtein",        "O(N × M)",
             "Edit distance — minimum operations to transform one text to another.",
             "#14b8a6"),
            ("LCS Algorithm",      "O(N × M)",
             "Longest Common Subsequence for sequential similarity detection.",
             "#f43f5e"),
            ("Word Overlap",       "O(N)",
             "Simple but effective word-level overlap detection.",
             "#a855f7"),
        ]:
            st.markdown(f"""
            <div class='algo-card'>
              <div style='display:flex;justify-content:space-between;
                align-items:center;margin-bottom:6px'>
                <span style='color:{color};font-weight:700'>
                  {name}
                </span>
                <span style='background:#1a2744;color:#8892b0;
                  padding:2px 8px;border-radius:10px;
                  font-size:0.75rem;font-family:monospace'>
                  {complexity}
                </span>
              </div>
              <span style='color:#8892b0;font-size:0.85rem'>
                {desc}
              </span>
            </div>
            """, unsafe_allow_html=True)

    # Workflow diagram
    st.markdown("---")
    st.markdown('<div class="section-title">🔄 Detection Workflow</div>',
                unsafe_allow_html=True)

    steps = [
        ("📄", "Input Documents",    "Original + Submitted text"),
        ("🧹", "Preprocessing",      "Clean, tokenize, normalize"),
        ("🔍", "String Matching",    "KMP + Rabin-Karp algorithms"),
        ("🔑", "Fingerprinting",     "Winnowing document hashes"),
        ("📊", "Similarity Metrics", "Jaccard, Cosine, LCS, Edit"),
        ("📈", "Score Calculation",  "Weighted average of all scores"),
        ("📄", "Final Report",       "Verdict + matched phrases"),
    ]

    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='text-align:center;
              background:#0a0f1e;border:1px solid #1a2744;
              border-radius:12px;padding:16px 8px'>
              <div style='font-size:1.8rem'>{icon}</div>
              <div style='color:#ccd6f6;font-weight:700;
                font-size:0.8rem;margin:6px 0 4px'>
                {title}
              </div>
              <div style='color:#8892b0;font-size:0.7rem'>
                {desc}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DETECT PLAGIARISM
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Detect Plagiarism":
    st.markdown("""
    <div class='hero-card'>
      <h2 style='margin:0;color:#8b5cf6'>🔍 Plagiarism Detector</h2>
      <p style='color:#8892b0;margin:5px 0 0'>
        Compare two documents using all algorithms
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Input method
    input_method = st.radio(
        "Input Method",
        ["📝 Type/Paste Text", "📂 Use Sample Files"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)

    if input_method == "📂 Use Sample Files":
        with col1:
            st.markdown("### 📄 Original Document")
            doc1_file = st.selectbox(
                "Select original",
                ["original.txt", "doc1.txt", "doc2.txt"]
            )
            text1 = load_sample(doc1_file)
            st.text_area(
                "Content preview",
                text1[:500] + "..." if len(text1) > 500
                else text1,
                height=200, disabled=True,
            )

        with col2:
            st.markdown("### 📄 Submitted Document")
            doc2_file = st.selectbox(
                "Select submitted",
                ["submitted.txt","doc2.txt","doc3.txt"]
            )
            text2 = load_sample(doc2_file)
            st.text_area(
                "Content preview",
                text2[:500] + "..." if len(text2) > 500
                else text2,
                height=200, disabled=True,
            )
        doc1_name = doc1_file
        doc2_name = doc2_file

    else:
        with col1:
            st.markdown("### 📄 Original Document")
            text1 = st.text_area(
                "Paste original text here",
                placeholder="Enter the original document text...",
                height=250,
            )
            doc1_name = "Document 1"

        with col2:
            st.markdown("### 📄 Submitted Document")
            text2 = st.text_area(
                "Paste submitted text here",
                placeholder="Enter the submitted document text...",
                height=250,
            )
            doc2_name = "Document 2"

    detect_btn = st.button("🔍 Detect Plagiarism Now!")

    if detect_btn:
        if not text1 or not text2:
            st.error("❌ Please provide both documents!")
        elif len(text1.split()) < 10 or \
             len(text2.split()) < 10:
            st.warning(
                "⚠️ Documents too short! "
                "Please provide at least 10 words each."
            )
        else:
            with st.spinner(
                "🔍 Running all algorithms..."
            ):
                detector = PlagiarismDetector()
                results  = detector.detect(
                    text1, text2,
                    doc1_name, doc2_name
                )
                generate_all_reports(results)

            score   = results["final_score"]
            vclass  = get_verdict_class(score)
            vcolor  = get_verdict_color(score)
            verdict = get_verdict_text(score)

            st.markdown("---")

            # Verdict
            st.markdown(f"""
            <div class='{vclass}'>
              <div style='font-size:3rem'>
                {'🔴' if score>=70 else '🟡' if score>=40
                 else '🟠' if score>=15 else '🟢'}
              </div>
              <div style='font-size:2rem;font-weight:900;
                color:{vcolor};margin:10px 0'>
                {score}% Plagiarism
              </div>
              <div style='font-size:1.1rem;
                color:{vcolor};font-weight:700'>
                {verdict}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("&nbsp;")

            # KPI metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🔍 KMP",
                      f"{results['kmp']['similarity']}%")
            c2.metric("🔢 Rabin-Karp",
                      f"{results['rabin_karp']['similarity']}%")
            c3.metric("🔑 Winnowing",
                      f"{results['winnowing']['similarity']}%")
            c4.metric("📐 Cosine",
                      f"{results['similarity']['cosine']}%")

            st.markdown("---")

            col_a, col_b = st.columns(2)

            with col_a:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": "Plagiarism Score (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": vcolor},
                        "steps": [
                            {"range":[0,15],
                             "color":"#052e16"},
                            {"range":[15,40],
                             "color":"#1c1917"},
                            {"range":[40,70],
                             "color":"#451a03"},
                            {"range":[70,100],
                             "color":"#450a0a"},
                        ],
                        "threshold": {
                            "line": {"color":"white",
                                     "width":3},
                            "thickness":0.8,
                            "value": 40,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    **PT, height=300,
                    margin=dict(t=40,b=10)
                )
                st.plotly_chart(
                    fig_gauge, use_container_width=True
                )

            with col_b:
                # Algorithm scores radar
                algo_scores = results["algorithm_scores"]
                categories  = list(algo_scores.keys())
                values      = list(algo_scores.values())

                fig_radar = go.Figure(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor="rgba(139,92,246,0.15)",
                    line=dict(color="#8b5cf6", width=2),
                    marker=dict(color="#8b5cf6", size=8),
                ))
                fig_radar.update_layout(
                    **PT, height=300,
                    polar=dict(
                        bgcolor="rgba(10,15,30,0.8)",
                        radialaxis=dict(
                            range=[0,100],
                            gridcolor="#1a2744",
                        ),
                        angularaxis=dict(
                            gridcolor="#1a2744",
                            color="#ccd6f6",
                        ),
                    ),
                    title=dict(
                        text="Algorithm Scores Radar",
                        font=dict(color="white",size=13)
                    ),
                    margin=dict(t=50,b=10),
                )
                st.plotly_chart(
                    fig_radar, use_container_width=True
                )

            # Matched phrases
            st.markdown(
                '<div class="section-title">'
                '📝 Matched Phrases (KMP)</div>',
                unsafe_allow_html=True
            )
            kmp_matches = results["kmp"]["matches"]
            if kmp_matches:
                for i, match in enumerate(
                    kmp_matches[:8], 1
                ):
                    phrase = match["phrase"]
                    if len(phrase) > 100:
                        phrase = phrase[:100] + "..."
                    st.markdown(f"""
                    <div class='match-card'>
                      <div style='color:#8b5cf6;
                        font-size:0.75rem;
                        font-weight:700;
                        margin-bottom:4px'>
                        Match #{i} •
                        {match['length']} words
                      </div>
                      <div style='color:#ccd6f6;
                        font-size:0.9rem'>
                        "{phrase}"
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No exact phrase matches found.")

            # Document stats
            st.markdown(
                '<div class="section-title">'
                '📊 Document Statistics</div>',
                unsafe_allow_html=True
            )
            stats_df = pd.DataFrame([
                results["doc1"],
                results["doc2"],
            ])
            st.dataframe(
                stats_df, use_container_width=True
            )

            st.success(
                "✅ Reports saved to outputs/ folder!"
            )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ALGORITHM ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Algorithm Analysis":
    st.markdown("""
    <div class='hero-card'>
      <h2 style='margin:0;color:#8b5cf6'>
        📊 Algorithm Analysis
      </h2>
      <p style='color:#8892b0;margin:5px 0 0'>
        Deep dive into algorithm performance and comparison
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Load sample for demo
    text1 = load_sample("original.txt")
    text2 = load_sample("submitted.txt")

    if text1 and text2:
        detector = PlagiarismDetector()
        results  = detector.detect(
            text1, text2,
            "original.txt", "submitted.txt"
        )

        algo_scores = results["algorithm_scores"]
        algos   = list(algo_scores.keys())
        scores  = list(algo_scores.values())
        colors  = [
            "#8b5cf6","#ec4899","#f59e0b","#10b981",
            "#3b82f6","#14b8a6","#f43f5e","#a855f7",
        ]

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart
            st.markdown(
                '<div class="section-title">'
                '📊 Algorithm Scores</div>',
                unsafe_allow_html=True
            )
            fig_bar = go.Figure(go.Bar(
                x=scores, y=algos,
                orientation="h",
                marker=dict(
                    color=colors[:len(algos)],
                    line=dict(color="#020408",width=1)
                ),
                text=[f"{s:.1f}%" for s in scores],
                textposition="outside",
            ))
            fig_bar.add_vline(
                x=40, line_dash="dash",
                line_color="#f59e0b",
                annotation_text="Threshold"
            )
            fig_bar.update_layout(
                **PT, height=380,
                xaxis=dict(
                    title="Score (%)",
                    range=[0,110],
                ),
                margin=dict(t=20,b=20,l=20,r=60)
            )
            st.plotly_chart(
                fig_bar, use_container_width=True
            )

        with col2:
            # Donut chart
            st.markdown(
                '<div class="section-title">'
                '🥧 Score Distribution</div>',
                unsafe_allow_html=True
            )
            fig_pie = go.Figure(go.Pie(
                labels=algos,
                values=scores,
                hole=0.55,
                marker=dict(
                    colors=colors[:len(algos)],
                    line=dict(color="#020408",width=2)
                ),
                textinfo="label+percent",
                textfont=dict(size=10),
            ))
            fig_pie.add_annotation(
                text=f"<b>{results['final_score']}%</b>"
                     "<br>Overall",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16,color="#8b5cf6")
            )
            fig_pie.update_layout(
                **PT, height=380,
                showlegend=False,
                margin=dict(t=20,b=20,l=10,r=10)
            )
            st.plotly_chart(
                fig_pie, use_container_width=True
            )

        # Complexity comparison
        st.markdown("---")
        st.markdown(
            '<div class="section-title">'
            '⚡ Algorithm Complexity Comparison</div>',
            unsafe_allow_html=True
        )

        complexity_data = {
            "Algorithm":  [
                "KMP","Rabin-Karp","Winnowing",
                "Jaccard","Cosine","Levenshtein",
                "LCS","Word Overlap"
            ],
            "Time Complexity": [
                "O(N+M)","O(N+M)","O(N)",
                "O(N)","O(N)","O(N×M)",
                "O(N×M)","O(N)"
            ],
            "Space Complexity": [
                "O(M)","O(1)","O(N)",
                "O(N)","O(N)","O(N×M)",
                "O(N×M)","O(N)"
            ],
            "Best For": [
                "Exact phrase","Multi-pattern",
                "Fingerprinting","Set similarity",
                "Doc similarity","Short text",
                "Sequential","Quick check"
            ],
            "Score (%)": scores,
        }
        df_comp = pd.DataFrame(complexity_data)
        st.dataframe(df_comp, use_container_width=True)

        # Similarity metrics detail
        st.markdown("---")
        st.markdown(
            '<div class="section-title">'
            '📐 Similarity Metrics Detail</div>',
            unsafe_allow_html=True
        )
        sim = results["similarity"]
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown(f"""
            <div class='algo-card'>
              <div style='color:#8b5cf6;font-weight:700;
                margin-bottom:8px'>
                📐 Cosine Similarity
              </div>
              <div style='color:#ccd6f6;font-size:2rem;
                font-weight:900'>{sim['cosine']}%</div>
              <div style='color:#8892b0;font-size:0.8rem;
                margin-top:6px'>
                TF vector dot product similarity
              </div>
            </div>
            <div class='algo-card' style='margin-top:8px'>
              <div style='color:#ec4899;font-weight:700;
                margin-bottom:8px'>
                🔗 Jaccard Similarity
              </div>
              <div style='color:#ccd6f6;font-size:2rem;
                font-weight:900'>{sim['jaccard']}%</div>
              <div style='color:#8892b0;font-size:0.8rem;
                margin-top:6px'>
                Shingle set intersection over union
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class='algo-card'>
              <div style='color:#f59e0b;font-weight:700;
                margin-bottom:8px'>
                📝 Levenshtein Distance
              </div>
              <div style='color:#ccd6f6;font-size:2rem;
                font-weight:900'>{sim['levenshtein']}%</div>
              <div style='color:#8892b0;font-size:0.8rem;
                margin-top:6px'>
                Edit distance: {sim['lev_distance']} ops
              </div>
            </div>
            <div class='algo-card' style='margin-top:8px'>
              <div style='color:#10b981;font-weight:700;
                margin-bottom:8px'>
                🔤 LCS Similarity
              </div>
              <div style='color:#ccd6f6;font-size:2rem;
                font-weight:900'>{sim['lcs']}%</div>
              <div style='color:#8892b0;font-size:0.8rem;
                margin-top:6px'>
                LCS length: {sim['lcs_length']} chars
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            st.markdown(f"""
            <div class='algo-card'>
              <div style='color:#3b82f6;font-weight:700;
                margin-bottom:8px'>
                💬 Word Overlap
              </div>
              <div style='color:#ccd6f6;font-size:2rem;
                font-weight:900'>{sim['word_overlap']}%</div>
              <div style='color:#8892b0;font-size:0.8rem;
                margin-top:6px'>
                Common words / total unique words
              </div>
            </div>
            <div class='algo-card' style='margin-top:8px'>
              <div style='color:#a855f7;font-weight:700;
                margin-bottom:8px'>
                ⭐ Overall Score
              </div>
              <div style='color:#ccd6f6;font-size:2rem;
                font-weight:900'>
                {results['final_score']}%
              </div>
              <div style='color:#8892b0;font-size:0.8rem;
                margin-top:6px'>
                Weighted average of all algorithms
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MULTI-DOCUMENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📂 Multi-Document":
    st.markdown("""
    <div class='hero-card'>
      <h2 style='margin:0;color:#8b5cf6'>
        📂 Multi-Document Detection
      </h2>
      <p style='color:#8892b0;margin:5px 0 0'>
        Compare one document against multiple documents
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    original = load_sample("original.txt")

    docs = []
    for fname in ["submitted.txt","doc1.txt",
                  "doc2.txt","doc3.txt"]:
        text = load_sample(fname)
        if text:
            docs.append((fname, text))

    if st.button("🔍 Run Multi-Document Detection"):
        detector = PlagiarismDetector()

        with st.spinner("Analyzing all documents..."):
            results_list = detector.detect_multiple(
                original, docs
            )

        # Results table
        st.markdown(
            '<div class="section-title">'
            '📊 Results Summary</div>',
            unsafe_allow_html=True
        )

        for r in results_list:
            color  = get_verdict_color(r["score"])
            vclass = get_verdict_class(r["score"])
            bar_w  = int(r["score"])

            st.markdown(f"""
            <div style='background:#0a0f1e;
              border:1px solid #1a2744;
              border-left:4px solid {color};
              border-radius:12px;padding:16px;
              margin:8px 0'>
              <div style='display:flex;
                justify-content:space-between;
                align-items:center;margin-bottom:8px'>
                <span style='color:#ccd6f6;
                  font-weight:700'>{r['name']}</span>
                <span style='color:{color};
                  font-weight:900;font-size:1.2rem'>
                  {r['score']}%
                </span>
              </div>
              <div style='background:#1a2744;
                border-radius:4px;height:6px;
                margin-bottom:8px'>
                <div style='background:{color};
                  width:{bar_w}%;height:6px;
                  border-radius:4px'></div>
              </div>
              <div style='color:{color};
                font-size:0.85rem;font-weight:600'>
                {r['verdict']}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Bar chart comparison
        st.markdown(
            '<div class="section-title">'
            '📊 Score Comparison</div>',
            unsafe_allow_html=True
        )
        names  = [r["name"]  for r in results_list]
        scores = [r["score"] for r in results_list]
        bcolors = [
            get_verdict_color(s) for s in scores
        ]

        fig = go.Figure(go.Bar(
            x=names, y=scores,
            marker=dict(
                color=bcolors,
                line=dict(color="#020408",width=1)
            ),
            text=[f"{s}%" for s in scores],
            textposition="outside",
        ))
        fig.add_hline(
            y=40, line_dash="dash",
            line_color="#f59e0b",
            annotation_text="Plagiarism Threshold (40%)"
        )
        fig.add_hline(
            y=70, line_dash="dash",
            line_color="#ef4444",
            annotation_text="High Risk (70%)"
        )
        fig.update_layout(
            **PT, height=380,
            yaxis=dict(
                title="Plagiarism Score (%)",
                range=[0,115],
            ),
            margin=dict(t=20,b=20,l=20,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📄 Reports":
    st.markdown("""
    <div class='hero-card'>
      <h2 style='margin:0;color:#8b5cf6'>📄 Reports</h2>
      <p style='color:#8892b0;margin:5px 0 0'>
        View and download generated reports
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Check for existing reports
    report_files = {
        "📝 Text Report":  "outputs/report.txt",
        "📊 CSV Report":   "outputs/report.csv",
        "🔢 JSON Report":  "outputs/report.json",
    }

    any_found = False
    for label, path in report_files.items():
        if os.path.exists(path):
            any_found = True

    if not any_found:
        st.info(
            "No reports yet! Go to "
            "**🔍 Detect Plagiarism** and run a detection first."
        )
    else:
        # Text report
        if os.path.exists("outputs/report.txt"):
            st.markdown(
                '<div class="section-title">'
                '📝 Text Report</div>',
                unsafe_allow_html=True
            )
            with open("outputs/report.txt",
                      encoding="utf-8") as f:
                report_text = f.read()
            st.code(report_text, language="text")

            st.download_button(
                "⬇️ Download Text Report",
                report_text,
                file_name="plagiarism_report.txt",
                mime="text/plain",
            )

        # CSV report
        if os.path.exists("outputs/report.csv"):
            st.markdown("---")
            st.markdown(
                '<div class="section-title">'
                '📊 CSV Report</div>',
                unsafe_allow_html=True
            )
            df = pd.read_csv("outputs/report.csv")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV Report",
                csv_data,
                file_name="plagiarism_report.csv",
                mime="text/csv",
            )

        # JSON report
        if os.path.exists("outputs/report.json"):
            st.markdown("---")
            st.markdown(
                '<div class="section-title">'
                '🔢 JSON Report</div>',
                unsafe_allow_html=True
            )
            import json
            with open("outputs/report.json",
                      encoding="utf-8") as f:
                json_data = json.load(f)
            st.json(json_data)

            st.download_button(
                "⬇️ Download JSON Report",
                json.dumps(json_data, indent=2),
                file_name="plagiarism_report.json",
                mime="application/json",
            )