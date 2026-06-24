# app/streamlit_app.py
import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioMed RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
# One font (Inter), one type scale, one theme. No per-element !important soup.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f6f7f9;
    --surface: #ffffff;
    --border: #e6e8ec;
    --text-primary: #1a1d23;
    --text-secondary: #565b66;
    --text-muted: #8b909b;
    --accent: #2563eb;
    --accent-light: #eef4ff;
    --accent-border: #cdddff;
    --green: #15a34a;
    --orange: #ea580c;
    --teal: #0e9bb8;
    --sidebar-bg: #15171c;
    --sidebar-text: #c3c7cf;
    --sidebar-muted: #6b7080;
    --radius: 12px;
}

/* ---- Base typography: everything inherits one family + size ---- */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp { background: var(--bg); }
.block-container { padding-top: 2.2rem; max-width: 1080px; }

/* Kill Streamlit's default top toolbar/footer clutter */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background: var(--sidebar-bg); }
section[data-testid="stSidebar"] * { color: var(--sidebar-text); }
.sb-brand { font-size: 1.15rem; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.sb-tagline { font-size: 0.8rem; color: var(--sidebar-muted); margin-top: 2px; }
.sb-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--sidebar-muted); margin: 0 0 0.6rem;
}
.sb-rule { height: 1px; background: #2a2d35; margin: 1.4rem 0 1.1rem; }
.sb-row { display: flex; align-items: center; gap: 9px; margin: 8px 0; font-size: 0.9rem; }
.sb-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.sb-foot { font-size: 0.78rem; color: var(--sidebar-muted); line-height: 1.9; }

/* ---- Header ---- */
.app-header {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.8rem 2rem; margin-bottom: 1.6rem;
}
.app-title { font-size: 1.7rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; margin: 0; }
.app-subtitle { font-size: 0.95rem; color: var(--text-secondary); margin: 0.4rem 0 0; }
.tag-row { margin-top: 1rem; display: flex; gap: 8px; flex-wrap: wrap; }
.ctag {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.78rem; font-weight: 600; color: var(--text-secondary);
    background: var(--bg); border: 1px solid var(--border);
    padding: 5px 12px; border-radius: 20px;
}
.ctag .sb-dot { width: 8px; height: 8px; }

/* ---- Tabs: make them unmistakably tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem; padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    height: auto; padding: 0.7rem 1.3rem; background: transparent;
    border: none; border-bottom: 2px solid transparent; border-radius: 8px 8px 0 0;
    font-size: 0.95rem; font-weight: 600; color: var(--text-muted);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-secondary); background: var(--bg); }
.stTabs [aria-selected="true"] {
    color: var(--accent); border-bottom: 2px solid var(--accent); background: var(--accent-light);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---- Section labels ---- */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.7rem;
}
.divider { height: 1px; background: var(--border); margin: 1.6rem 0; }

/* ---- Example query cards (the colored tag sits above each button) ---- */
.ex-tag { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; margin: 0 0 6px 2px; }

/* ---- Chat bubbles ---- */
.msg-role { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-muted); margin: 1rem 0 0.35rem; }
.user-bubble {
    background: var(--accent); color: #fff; padding: 0.9rem 1.2rem;
    border-radius: 14px 14px 4px 14px; font-size: 0.95rem; line-height: 1.6;
    margin-left: 18%;
}
.assistant-bubble {
    background: var(--surface); color: var(--text-primary); border: 1px solid var(--border);
    padding: 1.1rem 1.3rem; border-radius: 14px 14px 14px 4px;
    font-size: 0.95rem; line-height: 1.75; margin-right: 8%;
}

/* ---- Transparency panel ---- */
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.1rem 1.3rem; margin: 0.7rem 8% 0.7rem 0; }
.panel-title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.7rem; }
.panel-row { display: flex; align-items: center; gap: 0.7rem; margin: 6px 0; font-size: 0.88rem; }
.panel-key { font-weight: 600; color: var(--text-muted); min-width: 90px; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 7px; font-size: 0.78rem; font-weight: 600; margin-right: 4px; }
.chip-blue  { background: var(--accent-light); color: var(--accent); border: 1px solid var(--accent-border); }
.chip-green { background: #effaf2; color: var(--green); border: 1px solid #c4eccf; }
.chip-teal  { background: #e9fafd; color: var(--teal); border: 1px solid #bdebf2; }
.chip-orange{ background: #fff2ea; color: var(--orange); border: 1px solid #ffd6bd; }
.chip-gray  { background: var(--bg); color: var(--text-secondary); border: 1px solid var(--border); }

/* ---- Source cards ---- */
.source-card { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.4rem 0; font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6; }
.pubmed-card { background: #f4fcfe; border: 1px solid #bdebf2; border-left: 3px solid var(--teal); border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0.4rem 0; font-size: 0.88rem; color: #0c5566; line-height: 1.6; }
.source-meta { font-size: 0.76rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 500; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 7px; font-size: 0.7rem; font-weight: 600; margin-right: 5px; }
.pill-score { background: #effaf2; border: 1px solid #c4eccf; color: var(--green); }
.pill-sec   { background: var(--accent-light); border: 1px solid var(--accent-border); color: var(--accent); }
.pill-pmid  { background: #e9fafd; border: 1px solid #bdebf2; color: var(--teal); }

/* ---- Metric cards (dashboard) ---- */
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.3rem 1rem; text-align: center; }
.metric-value { font-size: 2.3rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
.metric-label { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.metric-delta { font-size: 0.74rem; color: var(--green); font-weight: 600; margin-top: 0.3rem; }

/* ---- Buttons ---- */
.stButton > button {
    background: var(--surface); color: var(--text-primary); border: 1px solid var(--border);
    border-radius: 10px; font-weight: 500; font-size: 0.9rem; padding: 0.85rem 1rem;
    text-align: left; width: 100%; white-space: normal; line-height: 1.5; height: 100%;
    transition: all 0.12s ease;
}
.stButton > button:hover { border-color: var(--accent); background: var(--accent-light); color: var(--accent); }
.stTextInput > div > div > input { border-radius: 10px; border: 1.5px solid var(--border); font-size: 0.92rem; }
.stTextInput > div > div > input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
CANCER_COLORS = {
    "Thyroid_Cancer": "#0e9bb8",
    "Lung_Cancer":    "#ea580c",
    "Colon_Cancer":   "#15a34a",
}

EXAMPLE_QUERIES = [
    ("Lung Cancer",    "What mechanisms contribute to acquired resistance to EGFR inhibitors in lung cancer?"),
    ("Colon Cancer",   "How do KRAS mutations influence response to EGFR-targeted therapy in colorectal cancer?"),
    ("Thyroid Cancer", "What is the role of BRAF mutations in papillary thyroid carcinoma?"),
    ("Colon Cancer",   "Why are microsatellite stable colorectal cancers less responsive to checkpoint blockade?"),
    ("Lung Cancer",    "What clinical evidence exists for MEK inhibitors in KRAS-mutant NSCLC?"),
    ("Thyroid Cancer", "Which immune checkpoints are targeted in thyroid cancer therapies?"),
]

# map a display name ("Lung Cancer") to its color
NAME_TO_COLOR = {k.replace("_", " "): v for k, v in CANCER_COLORS.items()}

RAGAS_CSV = Path("eval/runs/ragas_results.csv")
BASELINE  = {"faithfulness": 0.64, "answer_relevancy": 0.57, "context_utilization": 0.47}


# ── Lazy loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading agent…")
def load_agent():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.agent import run_agent
    return run_agent


def run_query(question: str):
    """Run the agent and append a user+assistant message pair to the chat."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching literature and generating answer…"):
        try:
            run_agent = load_agent()
            result    = run_agent(question)
            from app.query_router import infer_intent, extract_entities, extract_cancer_type
            intent      = infer_intent(question)
            entities    = extract_entities(question)
            cancer_type = extract_cancer_type(question)
            st.session_state.messages.append({
                "role":        "assistant",
                "content":     result["answer"],
                "sources":     result["local_chunks"],
                "pubmed":      result["pubmed_articles"],
                "used_pubmed": result["used_pubmed"],
                "reasoning": {
                    "intent":      intent,
                    "cancer":      cancer_type.replace("_", " ") if cancer_type else "General",
                    "entities":    entities[:6],
                    "used_pubmed": result["used_pubmed"],
                },
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}\n\nMake sure `OPENAI_API_KEY` is set and the FAISS index exists at `outputs/index_openai/`.",
                "sources": [], "pubmed": [], "reasoning": None,
            })


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div class='sb-brand'>BioMed RAG</div>"
        "<div class='sb-tagline'>Oncology Q&A System</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sb-rule'></div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-label'>Cancer Types</div>", unsafe_allow_html=True)
    for ct, color in CANCER_COLORS.items():
        st.markdown(
            f"<div class='sb-row'><span class='sb-dot' style='background:{color}'></span>"
            f"{ct.replace('_',' ')}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sb-rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-label'>Settings</div>", unsafe_allow_html=True)
    top_k          = st.slider("Top-K chunks", min_value=3, max_value=15, value=8, step=1)
    show_sources   = st.toggle("Show source chunks", value=True)
    show_reasoning = st.toggle("Show query reasoning", value=True)

    st.markdown("<div class='sb-rule'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sb-foot'>Hybrid FAISS + BM25 retrieval<br>"
        "Intent routing · Entity reranking<br>Cancer-type hard gating<br>"
        "LangGraph agent + PubMed fallback<br>Evaluation: RAGAS</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
tag_html = "".join(
    f"<span class='ctag'><span class='sb-dot' style='background:{c}'></span>{n}</span>"
    for n, c in NAME_TO_COLOR.items()
)
st.markdown(
    "<div class='app-header'>"
    "<div class='app-title'>Biomedical Research Assistant</div>"
    "<div class='app-subtitle'>Grounded answers from peer-reviewed oncology literature · "
    "Hybrid RAG with live PubMed fallback</div>"
    f"<div class='tag-row'>{tag_html}</div>"
    "</div>",
    unsafe_allow_html=True,
)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Research Chat", "Evaluation Dashboard"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Example query cards ───────────────────────────────────────────────
    st.markdown("<div class='section-label'>Try an example query</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (cancer, query) in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 3]:
            color = NAME_TO_COLOR.get(cancer, "#8b909b")
            st.markdown(f"<div class='ex-tag' style='color:{color}'>{cancer.upper()}</div>", unsafe_allow_html=True)
            if st.button(query, key=f"ex_{idx}"):
                run_query(query)
                st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Conversation ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown("<div class='msg-role' style='text-align:right'>You</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='msg-role'>Assistant</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

            # Transparency panel
            if show_reasoning and msg.get("reasoning"):
                r = msg["reasoning"]
                intent_chip = f"<span class='chip chip-blue'>{r.get('intent','—')}</span>"
                cancer_chip = f"<span class='chip chip-orange'>{r.get('cancer','General')}</span>"
                source_chip = f"<span class='chip chip-teal'>{'PubMed + Local' if r.get('used_pubmed') else 'Local Index'}</span>"
                entity_chips = "".join(f"<span class='chip chip-gray'>{e}</span>" for e in r.get('entities', [])[:6]) \
                               or "<span class='chip chip-gray'>none detected</span>"
                st.markdown(
                    "<div class='panel'>"
                    "<div class='panel-title'>How this answer was generated</div>"
                    f"<div class='panel-row'><span class='panel-key'>Intent</span>{intent_chip}</div>"
                    f"<div class='panel-row'><span class='panel-key'>Cancer type</span>{cancer_chip}</div>"
                    f"<div class='panel-row'><span class='panel-key'>Source</span>{source_chip}</div>"
                    f"<div class='panel-row'><span class='panel-key'>Entities</span>{entity_chips}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Sources
            if show_sources:
                local_chunks    = msg.get("sources", [])
                pubmed_articles = msg.get("pubmed", [])
                total = len(local_chunks) + len(pubmed_articles)
                if total > 0:
                    lbl = f"{len(local_chunks)} local chunks"
                    if pubmed_articles:
                        lbl += f"  ·  {len(pubmed_articles)} PubMed abstracts"
                    with st.expander(lbl):
                        if local_chunks:
                            st.markdown("<div class='section-label'>Local corpus chunks</div>", unsafe_allow_html=True)
                            for src in local_chunks[:5]:
                                score   = src.get("score", 0)
                                section = src.get("section", "—")
                                paper   = src.get("paper_id", "—")
                                text    = (src.get("text") or "")[:350].replace("\n", " ")
                                st.markdown(
                                    "<div class='source-card'>"
                                    "<div class='source-meta'>"
                                    f"<span class='pill pill-score'>↑ {score:.3f}</span>"
                                    f"<span class='pill pill-sec'>{section}</span>&nbsp;{paper}"
                                    f"</div>{text}…</div>",
                                    unsafe_allow_html=True,
                                )
                        if pubmed_articles:
                            st.markdown("<div class='section-label' style='margin-top:0.8rem'>Live PubMed abstracts</div>", unsafe_allow_html=True)
                            for a in pubmed_articles:
                                st.markdown(
                                    "<div class='pubmed-card'>"
                                    "<div class='source-meta'>"
                                    f"<span class='pill pill-pmid'>PMID {a.get('pmid','?')}</span>&nbsp;{a.get('year','?')}"
                                    "</div>"
                                    f"<strong>{a.get('title','')}</strong><br>"
                                    f"<span style='color:#0c5566'>{a.get('abstract','')[:300]}…</span>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )

    # ── Input ─────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        pending    = st.session_state.pop("pending_query", "")
        user_input = st.text_input(
            "Question",
            value=pending,
            placeholder="e.g. How do BRAF mutations affect papillary thyroid carcinoma?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        send = st.button("Ask", use_container_width=True)

    if send and user_input.strip():
        run_query(user_input.strip())
        st.rerun()

    if st.session_state.messages:
        if st.button("Clear conversation", key="clear"):
            st.session_state.messages = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EVAL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("<div class='app-title' style='font-size:1.4rem'>RAGAS Evaluation Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle' style='margin-bottom:1.5rem'>30 queries · GPT-4o-mini · text-embedding-3-large</div>", unsafe_allow_html=True)

    if not RAGAS_CSV.exists():
        st.info("No evaluation results found. Run `make eval` to generate results.")
    else:
        df = pd.read_csv(RAGAS_CSV)
        score_cols = [c for c in df.columns if any(
            m in c.lower() for m in ["faithfulness", "answer_relevancy", "context_relevance",
                                      "relevancy", "relevance", "utilization"]
        )]

        # ── Metric cards ──────────────────────────────────────────────────
        if score_cols:
            metric_cols = st.columns(len(score_cols))
            for i, col_name in enumerate(score_cols):
                val      = df[col_name].mean()
                label    = col_name.replace("_", " ").title()
                color    = "#15a34a" if val >= 0.7 else "#d97706" if val >= 0.5 else "#dc2626"
                baseline = BASELINE.get(col_name)
                delta_html = ""
                if baseline:
                    pct  = (val - baseline) / baseline * 100
                    sign = "+" if pct >= 0 else ""
                    delta_html = f"<div class='metric-delta'>{sign}{pct:.0f}% from baseline</div>"
                with metric_cols[i]:
                    st.markdown(
                        "<div class='metric-card'>"
                        f"<div class='metric-value' style='color:{color}'>{val:.2f}</div>"
                        f"<div class='metric-label'>{label}</div>{delta_html}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ── Charts ────────────────────────────────────────────────────
            palette = ["#2563eb", "#ea580c", "#0e9bb8", "#a855f7"]
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("<div class='section-label'>Score Distributions</div>", unsafe_allow_html=True)
                fig = go.Figure()
                for i, col_name in enumerate(score_cols):
                    fig.add_trace(go.Violin(
                        y=df[col_name].dropna(),
                        name=col_name.replace("_", " ").title(),
                        box_visible=True, meanline_visible=True,
                        fillcolor=palette[i % len(palette)],
                        opacity=0.55, line_color=palette[i % len(palette)],
                    ))
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="Inter", size=11),
                    yaxis=dict(range=[0, 1], gridcolor="#eef0f3"),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_chart2:
                st.markdown("<div class='section-label'>Current vs Baseline</div>", unsafe_allow_html=True)
                metrics_display, current_vals, baseline_vals = [], [], []
                for col_name in score_cols:
                    metrics_display.append(col_name.replace("_", " ").title())
                    current_vals.append(df[col_name].mean())
                    baseline_vals.append(BASELINE.get(col_name, 0))

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    name="Baseline", x=metrics_display, y=baseline_vals,
                    marker_color="#dfe3e8", marker_line_width=0,
                ))
                fig2.add_trace(go.Bar(
                    name="Current", x=metrics_display, y=current_vals,
                    marker_color=palette[:len(metrics_display)], marker_line_width=0,
                    text=[f"{v:.3f}" for v in current_vals], textposition="outside",
                ))
                fig2.update_layout(
                    barmode="group", height=300, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="Inter", size=11),
                    yaxis=dict(range=[0, 1.15], gridcolor="#eef0f3"),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── Per-query table ───────────────────────────────────────────────
        st.markdown("<div class='section-label' style='margin-top:0.5rem'>Per-Query Scores</div>", unsafe_allow_html=True)
        display_df = df.copy()
        if "question" in display_df.columns:
            display_df["question"] = display_df["question"].str[:75] + "…"
        cols_to_show = (["question"] if "question" in display_df.columns else []) + score_cols
        st.dataframe(
            display_df[cols_to_show].style
            .format({c: "{:.3f}" for c in score_cols})
            .background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True, height=380,
        )
        st.download_button(
            "Download results CSV",
            data=df.to_csv(index=False),
            file_name="ragas_results.csv",
            mime="text/csv",
        )

    # ── Corpus stats ──────────────────────────────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Corpus & Pipeline Stats</div>", unsafe_allow_html=True)

    stats_path = Path("outputs/chunks_stats.json")
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, val, label in [
            (sc1, stats['total_docs'], "Papers"),
            (sc2, f"{stats['total_chunks']:,}", "Chunks"),
            (sc3, int(stats['avg_chunk_tokens']), "Avg Tokens/Chunk"),
            (sc4, "~7K→484", "Dedup Pipeline"),
        ]:
            with col:
                st.markdown(
                    "<div class='metric-card'>"
                    f"<div class='metric-value' style='font-size:1.7rem'>{val}</div>"
                    f"<div class='metric-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Run `make ingest` to populate corpus stats.")
