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

# Force font size via JavaScript
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    document.documentElement.style.fontSize = '18px';
});
</script>
""", unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────────


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

:root { font-size: 18px !important; }
* { box-sizing: border-box; }

:root {
    --bg: #f7f6f3;
    --surface: #ffffff;
    --border: #e5e3dc;
    --text-primary: #1a1916;
    --text-secondary: #6b6860;
    --text-muted: #9c9a96;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --accent-border: #bfdbfe;
    --green: #16a34a;
    --green-light: #f0fdf4;
    --green-border: #bbf7d0;
    --orange: #ea580c;
    --orange-light: #fff7ed;
    --teal: #0891b2;
    --teal-light: #ecfeff;
    --sidebar-bg: #1a1916;
    --sidebar-border: #2c2a26;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    font-size: 16px;
}

section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border);
}
section[data-testid="stSidebar"] * { color: #d4d0c8 !important; }

.app-header {
    background: var(--sidebar-bg);
    border: 1px solid #2c2a26;
    border-radius: 16px;
    padding: 2rem 2.2rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::after {
    content: '🧬';
    position: absolute;
    right: 1.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 4rem;
    opacity: 0.08;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #f5f3ee;
    margin: 0 0 0.3rem 0;
}
.app-subtitle { color: #6b6860; font-size: 0.9rem; font-weight: 300; margin: 0; }
.cancer-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #d4d0c8; font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    margin-right: 5px; margin-top: 0.8rem;
    letter-spacing: 0.5px; text-transform: uppercase;
}

.user-bubble {
    background: var(--sidebar-bg); color: #f5f3ee;
    padding: 1rem 1.4rem;
    font-size: 1rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.6rem 0 0.6rem 4rem;
    line-height: 1.6;
}
.assistant-bubble {
    background: var(--surface); color: var(--text-primary);
    padding: 1.3rem 1.6rem;
    font-size: 1.05rem;
    line-height: 1.85;
    border-radius: 16px 16px 16px 4px;
    margin: 0.6rem 4rem 0.4rem 0;
    border: 1px solid var(--border);
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}

.transparency-panel {
    background: #fafaf8; border: 1px solid var(--border);
    border-radius: 10px;
    margin: 0 4rem 0.8rem 0;
    font-size: 0.92rem;
    padding: 1rem 1.2rem;
}
.transparency-title {
    font-family: 'Syne', sans-serif; font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--text-muted); margin-bottom: 0.5rem;
}
.transparency-row { display: flex; align-items: center; gap: 0.5rem; margin: 3px 0; color: var(--text-secondary); }
.transparency-key { font-weight: 600; color: var(--text-muted); min-width: 80px; font-size: 0.88rem; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; margin-right: 3px; }
.tag-blue  { background: var(--accent-light); color: var(--accent); border: 1px solid var(--accent-border); }
.tag-green { background: var(--green-light);  color: var(--green);  border: 1px solid var(--green-border); }
.tag-teal  { background: var(--teal-light);   color: var(--teal);   border: 1px solid #a5f3fc; }
.tag-orange{ background: var(--orange-light); color: var(--orange); border: 1px solid #fed7aa; }
.tag-gray  { background: #f5f3ee; color: var(--text-muted); border: 1px solid var(--border); }

.source-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 10px;
    padding: 1rem 1.1rem; margin: 0.35rem 0;
    font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6;
}
.pubmed-card {
    background: var(--teal-light); border: 1px solid #a5f3fc;
    border-left: 3px solid var(--teal); border-radius: 10px;
    padding: 1rem 1.1rem; margin: 0.35rem 0;
    font-size: 0.9rem; color: #164e63; line-height: 1.6;
}
.source-meta { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.35rem; font-weight: 500; }
.score-pill {
    display: inline-block; background: var(--green-light);
    border: 1px solid var(--green-border); color: var(--green);
    font-size: 0.68rem; font-weight: 700; padding: 1px 7px; border-radius: 8px; margin-right: 5px;
}
.section-pill {
    display: inline-block; background: var(--accent-light);
    border: 1px solid var(--accent-border); color: var(--accent);
    font-size: 0.68rem; font-weight: 500; padding: 1px 7px; border-radius: 8px; margin-right: 5px;
}
.pmid-pill {
    display: inline-block; background: var(--teal-light);
    border: 1px solid #a5f3fc; color: var(--teal);
    font-size: 0.68rem; font-weight: 600; padding: 1px 7px; border-radius: 8px;
}

.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif; font-size: 2.4rem;
    font-weight: 800; color: var(--text-primary); line-height: 1;
}
.metric-label {
    font-size: 0.75rem; color: var(--text-muted); margin-top: 0.3rem;
    text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;
}
.metric-delta { font-size: 0.72rem; color: var(--green); font-weight: 600; margin-top: 0.25rem; }

.stTextInput > div > div > input {
    border-radius: 10px !important; border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.92rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}

.stButton > button {
    background: var(--sidebar-bg) !important; color: #f5f3ee !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 0.55rem 1.2rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { background: #2c2a26 !important; transform: translateY(-1px) !important; }

div[data-testid="column"] .stButton > button {
    background: var(--surface) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    text-align: left !important; width: 100% !important;
    padding: 0.85rem 1rem !important; border-radius: 10px !important;
    font-size: 0.93rem !important; height: auto !important;
    white-space: normal !important; line-height: 1.4 !important;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: var(--accent) !important; background: var(--accent-light) !important;
    color: var(--accent) !important; transform: none !important;
}

.divider { height: 1px; background: var(--border); margin: 1rem 0; }
.section-label {
    font-family: 'Syne', sans-serif; font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--text-muted); margin-bottom: 0.6rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #eeece8; padding: 4px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.87rem !important; font-weight: 500 !important; color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: var(--text-primary) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
            

/* ── Force font size overrides ── */
.stApp, .stApp p, .stApp div, .stApp span, .stApp label {
    font-size: 16px !important;
}

/* Chat bubbles - direct override */
.user-bubble, .assistant-bubble {
    font-size: 1.05rem !important;
    line-height: 1.85 !important;
}

/* Example buttons */
div[data-testid="column"] .stButton > button,
div[data-testid="column"] .stButton > button * {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    padding: 0.9rem 1rem !important;
}

/* Transparency panel */
.transparency-panel, .transparency-panel * {
    font-size: 0.92rem !important;
}
.transparency-title { font-size: 0.82rem !important; }

/* Source cards */
.source-card, .pubmed-card,
.source-card *, .pubmed-card * {
    font-size: 0.9rem !important;
}

/* Section labels */
.section-label { font-size: 0.82rem !important; }

/* Sidebar text */
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p {
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
CANCER_COLORS = {
    "Thyroid_Cancer": "#0891b2",
    "Lung_Cancer":    "#ea580c",
    "Colon_Cancer":   "#16a34a",
}

EXAMPLE_QUERIES = [
    ("Lung Cancer",    "🫁", "What mechanisms contribute to acquired resistance to EGFR inhibitors in lung cancer?"),
    ("Colon Cancer",   "🫠", "How do KRAS mutations influence response to EGFR-targeted therapy in colorectal cancer?"),
    ("Thyroid Cancer", "🦋", "What is the role of BRAF mutations in papillary thyroid carcinoma?"),
    ("Colon Cancer",   "🫠", "Why are microsatellite stable colorectal cancers less responsive to checkpoint blockade?"),
    ("Lung Cancer",    "🫁", "What clinical evidence exists for MEK inhibitors in KRAS-mutant NSCLC?"),
    ("Thyroid Cancer", "🦋", "Which immune checkpoints are targeted in thyroid cancer therapies?"),
]

RAGAS_CSV   = Path("eval/runs/ragas_results.csv")
BASELINE    = {"faithfulness": 0.64, "answer_relevancy": 0.57, "context_utilization": 0.47}


# ── Lazy loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading agent…")
def load_agent():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.agent import run_agent
    return run_agent


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;"
        "color:#f5f3ee;margin:0 0 0.2rem 0'>🧬 BioMed RAG</p>"
        "<p style='font-size:0.75rem;color:#6b6860;margin:0 0 1rem 0'>Oncology Q&A System</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1px;background:#2c2a26;margin:0 0 1rem 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#6b6860;margin-bottom:0.5rem'>Cancer Types</p>", unsafe_allow_html=True)

    for ct, color in CANCER_COLORS.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0'></div>"
            f"<span style='font-size:0.83rem;color:#d4d0c8'>{ct.replace('_',' ')}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1px;background:#2c2a26;margin:1rem 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;color:#6b6860;margin-bottom:0.5rem'>Settings</p>", unsafe_allow_html=True)

    top_k          = st.slider("Top-K chunks", min_value=3, max_value=15, value=8, step=1)
    show_sources   = st.toggle("Show source chunks", value=True)
    show_reasoning = st.toggle("Show query reasoning", value=True)

    st.markdown("<div style='height:1px;background:#2c2a26;margin:1rem 0'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.73rem;color:#4a4840;line-height:1.8'>"
        "Hybrid FAISS + BM25 retrieval<br>Intent routing · Entity reranking<br>"
        "Cancer-type hard gating<br>LangGraph agent + PubMed fallback<br>Evaluation: RAGAS"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">Biomedical Research Assistant</p>
    <p class="app-subtitle">Grounded answers from peer-reviewed oncology literature · Hybrid RAG + live PubMed search</p>
    <div style="margin-top:0.8rem">
        <span class="cancer-badge">🫁 Lung Cancer</span>
        <span class="cancer-badge">🫠 Colon Cancer</span>
        <span class="cancer-badge">🦋 Thyroid Cancer</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬  Research Chat", "📊  Evaluation Dashboard"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Example query buttons ─────────────────────────────────────────────
    st.markdown("<div class='section-label'>Try an example query — click to load</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, (cancer, icon, query) in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 3]:
            short = query[:62] + "…" if len(query) > 62 else query
            btn_label = f"{icon} **{cancer}**\n{short}"
            if st.button(btn_label, key=f"ex_{idx}"):
                # Directly add to messages and process — bypass the input box
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("Searching literature and generating answer…"):
                    try:
                        run_agent = load_agent()
                        result    = run_agent(query)
                        from app.query_router import infer_intent, extract_entities, extract_cancer_type
                        intent      = infer_intent(query)
                        entities    = extract_entities(query)
                        cancer_type = extract_cancer_type(query)
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
                            "content": f"⚠️ Error: {e}",
                            "sources": [], "pubmed": [], "reasoning": None,
                        })
                st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Conversation ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

            # Transparency panel
            if show_reasoning and msg.get("reasoning"):
                r = msg["reasoning"]
                intent_tag  = f"<span class='tag tag-blue'>{r.get('intent','—')}</span>"
                cancer_tag  = f"<span class='tag tag-orange'>{r.get('cancer','General')}</span>"
                source_tag  = f"<span class='tag tag-teal'>{'🌐 PubMed + Local' if r.get('used_pubmed') else '🗂 Local Index'}</span>"
                entity_tags = "".join(f"<span class='tag tag-gray'>{e}</span>" for e in r.get('entities', [])[:6]) \
                              or "<span class='tag tag-gray'>none detected</span>"
                st.markdown(
                    f"<div class='transparency-panel'>"
                    f"<div class='transparency-title'>⚙ How this answer was generated</div>"
                    f"<div class='transparency-row'><span class='transparency-key'>Intent</span>{intent_tag}</div>"
                    f"<div class='transparency-row'><span class='transparency-key'>Cancer type</span>{cancer_tag}</div>"
                    f"<div class='transparency-row'><span class='transparency-key'>Source</span>{source_tag}</div>"
                    f"<div class='transparency-row'><span class='transparency-key'>Entities</span>{entity_tags}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Sources
            if show_sources:
                local_chunks    = msg.get("sources", [])
                pubmed_articles = msg.get("pubmed", [])
                total = len(local_chunks) + len(pubmed_articles)

                if total > 0:
                    lbl = f"📄 {len(local_chunks)} local chunks"
                    if pubmed_articles:
                        lbl += f" · 🌐 {len(pubmed_articles)} PubMed abstracts"
                    with st.expander(lbl):
                        if local_chunks:
                            st.markdown("<div class='section-label'>Local corpus chunks</div>", unsafe_allow_html=True)
                            for src in local_chunks[:5]:
                                score   = src.get("score", 0)
                                section = src.get("section", "—")
                                paper   = src.get("paper_id", "—")
                                text    = (src.get("text") or "")[:350].replace("\n", " ")
                                st.markdown(
                                    f"<div class='source-card'>"
                                    f"<div class='source-meta'>"
                                    f"<span class='score-pill'>↑ {score:.3f}</span>"
                                    f"<span class='section-pill'>{section}</span>&nbsp;{paper}"
                                    f"</div>{text}…</div>",
                                    unsafe_allow_html=True,
                                )
                        if pubmed_articles:
                            st.markdown("<div class='section-label' style='margin-top:0.8rem'>Live PubMed abstracts</div>", unsafe_allow_html=True)
                            for a in pubmed_articles:
                                st.markdown(
                                    f"<div class='pubmed-card'>"
                                    f"<div class='source-meta'>"
                                    f"<span class='pmid-pill'>PMID {a.get('pmid','?')}</span>&nbsp;{a.get('year','?')}"
                                    f"</div>"
                                    f"<strong style='font-size:0.83rem'>{a.get('title','')}</strong><br>"
                                    f"<span style='font-size:0.8rem;color:#164e63'>{a.get('abstract','')[:300]}…</span>"
                                    f"</div>",
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
        send = st.button("Ask →", use_container_width=True)

    if send and user_input.strip():
        question = user_input.strip()
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
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑 Clear conversation", key="clear"):
            st.session_state.messages = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EVAL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown(
        "<p style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;margin:0 0 0.2rem 0'>RAGAS Evaluation Results</p>"
        "<p style='color:#6b6860;font-size:0.88rem;margin:0 0 1.5rem 0'>30 queries · GPT-4o-mini · text-embedding-3-large</p>",
        unsafe_allow_html=True,
    )

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
                color    = "#16a34a" if val >= 0.7 else "#f59e0b" if val >= 0.5 else "#ef4444"
                baseline = BASELINE.get(col_name)
                delta_html = ""
                if baseline:
                    pct  = (val - baseline) / baseline * 100
                    sign = "+" if pct >= 0 else ""
                    delta_html = f"<div class='metric-delta'>{sign}{pct:.0f}% from baseline</div>"
                with metric_cols[i]:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value' style='color:{color}'>{val:.2f}</div>"
                        f"<div class='metric-label'>{label}</div>{delta_html}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ── Charts ────────────────────────────────────────────────────
            palette = ["#2563eb", "#ea580c", "#0891b2", "#a855f7"]
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
                    height=300, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="DM Sans", size=11),
                    yaxis=dict(range=[0,1], gridcolor="#f1f0ed"),
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
                    marker_color="#e5e3dc", marker_line_width=0,
                ))
                fig2.add_trace(go.Bar(
                    name="Current", x=metrics_display, y=current_vals,
                    marker_color=palette[:len(metrics_display)], marker_line_width=0,
                    text=[f"{v:.3f}" for v in current_vals], textposition="outside",
                ))
                fig2.update_layout(
                    barmode="group", height=300, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="DM Sans", size=11),
                    yaxis=dict(range=[0,1.15], gridcolor="#f1f0ed"),
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
            "⬇️ Download results CSV",
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
                    f"<div class='metric-card'>"
                    f"<div class='metric-value' style='font-size:1.8rem'>{val}</div>"
                    f"<div class='metric-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Run `make ingest` to populate corpus stats.")