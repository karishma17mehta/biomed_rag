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

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioMed RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Main background */
.main { background: #f8f9fc; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 60%, #0d2137 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Chat bubbles */
.user-bubble {
    background: #1e40af;
    color: white;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.95rem;
    line-height: 1.6;
}
.assistant-bubble {
    background: white;
    color: #1e293b;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Source cards */
.source-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.6;
}
.source-meta {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.score-pill {
    display: inline-block;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #16a34a;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 1px 8px;
    border-radius: 10px;
    margin-right: 6px;
}
.section-pill {
    display: inline-block;
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    color: #0369a1;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 1px 8px;
    border-radius: 10px;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #0f172a;
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}
.metric-delta {
    font-size: 0.75rem;
    color: #16a34a;
    font-weight: 600;
    margin-top: 0.2rem;
}

/* Input styling */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1e293b !important;
    transform: translateY(-1px) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 0.45rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #0f172a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

.divider { height: 1px; background: #e2e8f0; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── lazy imports (only load heavy stuff when needed) ─────────────────────────
@st.cache_resource(show_spinner="Loading retrieval index…")
def load_retriever():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.retrieve_faiss import retrieve
    return retrieve


@st.cache_resource(show_spinner="Loading answer generator…")
def load_generator():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm


# ── helpers ───────────────────────────────────────────────────────────────────
CANCER_COLORS = {
    "Thyroid_Cancer": "#0ea5e9",
    "Lung_Cancer":    "#f97316",
    "Colon_Cancer":   "#22c55e",
}

EXAMPLE_QUERIES = [
    ("Lung Cancer",    "What mechanisms contribute to acquired resistance to EGFR inhibitors?"),
    ("Colon Cancer",   "How do KRAS mutations influence response to EGFR-targeted therapy in colorectal cancer?"),
    ("Thyroid Cancer", "What targeted drugs are evaluated in differentiated thyroid cancer?"),
    ("Colon Cancer",   "Why are microsatellite stable colorectal cancers less responsive to checkpoint blockade?"),
    ("Lung Cancer",    "What clinical evidence exists for MEK inhibitors in KRAS-mutant NSCLC?"),
    ("Thyroid Cancer", "What resistance mechanisms are described for lenvatinib in thyroid cancer?"),
]

RAGAS_CSV = Path("eval/runs/ragas_results.csv")


def generate_answer(llm, question: str, contexts: list) -> str:
    """Grounded answer using retrieved contexts."""
    ctx_block = "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(contexts[:6])])
    prompt = f"""You are a biomedical assistant.
Use ONLY the provided contexts. Each claim must be supported.
If contexts lack evidence, say so explicitly.

QUESTION: {question}

CONTEXTS:
{ctx_block}

Write a concise, grounded answer (3-5 sentences). End each claim with a citation like [CTX 1]."""
    return llm.invoke(prompt).content.strip()


def cancer_color(text: str) -> str:
    for k, v in CANCER_COLORS.items():
        if k.lower().replace("_", " ") in text.lower():
            return v
    return "#94a3b8"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 BioMed RAG")
    st.markdown("<div style='height:1px;background:#1e2130;margin:0.5rem 0 1rem'></div>", unsafe_allow_html=True)

    st.markdown("**Cancer Types**")
    for ct, color in CANCER_COLORS.items():
        label = ct.replace("_", " ")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{color}'></div>"
            f"<span style='font-size:0.85rem'>{label}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1px;background:#1e2130;margin:1rem 0'></div>", unsafe_allow_html=True)

    st.markdown("**Retrieval Settings**")
    top_k = st.slider("Top-K chunks", min_value=3, max_value=15, value=8, step=1)
    show_sources = st.toggle("Show source chunks", value=True)

    st.markdown("<div style='height:1px;background:#1e2130;margin:1rem 0'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.75rem;color:#475569;line-height:1.6'>"
        "Hybrid FAISS + BM25 retrieval<br>"
        "Intent routing · Entity reranking<br>"
        "Cancer-type hard gating<br>"
        "Evaluation: RAGAS 0.4.x"
        "</div>",
        unsafe_allow_html=True
    )


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">🧬 Biomedical Research Assistant</p>
    <p class="app-subtitle">Grounded answers from peer-reviewed oncology literature</p>
    <div style="margin-top:1rem">
        <span class="badge">Thyroid Cancer</span>
        <span class="badge">Lung Cancer</span>
        <span class="badge">Colon Cancer</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬  Research Chat", "📊  Evaluation Dashboard"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # example queries
    st.markdown("**Try an example:**")
    cols = st.columns(3)
    for idx, (cancer, query) in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 3]:
            if st.button(f"🔬 {cancer}", key=f"ex_{idx}", help=query):
                st.session_state.pending_query = query

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # render conversation history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
            if show_sources and msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source chunks retrieved"):
                    for i, src in enumerate(msg["sources"], 1):
                        score = src.get("score", 0)
                        section = src.get("section", "—")
                        paper = src.get("paper_id", "—")
                        text_preview = (src.get("text") or "")[:350].replace("\n", " ")
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<div class='source-meta'>"
                            f"<span class='score-pill'>score {score:.3f}</span>"
                            f"<span class='section-pill'>{section}</span>"
                            f"&nbsp;·&nbsp;{paper}"
                            f"</div>"
                            f"{text_preview}…"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # input area
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Ask a biomedical research question",
            value=st.session_state.pop("pending_query", ""),
            placeholder="e.g. How do BRAF mutations affect papillary thyroid carcinoma?",
            label_visibility="collapsed",
            key="chat_input"
        )
    with col_btn:
        send = st.button("Ask →", use_container_width=True)

    if (send or user_input) and user_input.strip():
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Retrieving and generating answer…"):
            try:
                retrieve = load_retriever()
                llm = load_generator()

                hits = retrieve(question, top_k=top_k, max_per_paper=1, max_per_hash=1)
                contexts = [(h.get("text") or "")[:1500] for h in hits if h.get("text")]

                answer = generate_answer(llm, question, contexts)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": hits
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}\n\nMake sure `OPENAI_API_KEY` is set and the FAISS index exists at `outputs/index_openai/`.",
                    "sources": []
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

    st.markdown("### RAGAS Evaluation Results")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem'>Automated evaluation across 30 queries · "
        "GPT-4o-mini for generation · text-embedding-3-large for retrieval</p>",
        unsafe_allow_html=True
    )

    if not RAGAS_CSV.exists():
        st.info(
            "No evaluation results found yet. Run `python -m eval.06_run_ragas` "
            "to generate `eval/runs/ragas_results.csv`."
        )
    else:
        df = pd.read_csv(RAGAS_CSV)

        # ── metric columns detection ──────────────────────────────────────
        score_cols = [c for c in df.columns if any(
            m in c.lower() for m in ["faithfulness", "answer_relevancy", "context_relevance", "relevancy", "relevance"]
        )]

        # ── summary metric cards ──────────────────────────────────────────
        if score_cols:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            metric_cols = st.columns(len(score_cols))
            for i, col_name in enumerate(score_cols):
                val = df[col_name].mean()
                label = col_name.replace("_", " ").title()
                color = "#16a34a" if val >= 0.7 else "#f59e0b" if val >= 0.5 else "#ef4444"
                with metric_cols[i]:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value' style='color:{color}'>{val:.2f}</div>"
                        f"<div class='metric-label'>{label}</div>"
                        f"<div class='metric-delta'>avg across {len(df)} queries</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ── distribution chart ────────────────────────────────────────
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("**Score Distributions**")
                fig = go.Figure()
                palette = ["#0ea5e9", "#f97316", "#22c55e", "#a855f7"]
                for i, col_name in enumerate(score_cols):
                    fig.add_trace(go.Violin(
                        y=df[col_name].dropna(),
                        name=col_name.replace("_", " ").title(),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=palette[i % len(palette)],
                        opacity=0.6,
                        line_color=palette[i % len(palette)],
                    ))
                fig.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(family="DM Sans", size=11),
                    showlegend=True,
                    yaxis=dict(range=[0, 1], gridcolor="#f1f5f9"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_chart2:
                st.markdown("**Average Scores by Metric**")
                avg_scores = {col.replace("_", " ").title(): df[col].mean() for col in score_cols}
                fig2 = go.Figure(go.Bar(
                    x=list(avg_scores.keys()),
                    y=list(avg_scores.values()),
                    marker_color=["#0ea5e9", "#f97316", "#22c55e"][:len(avg_scores)],
                    marker_line_width=0,
                    text=[f"{v:.3f}" for v in avg_scores.values()],
                    textposition="outside",
                ))
                fig2.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(family="DM Sans", size=11),
                    yaxis=dict(range=[0, 1.1], gridcolor="#f1f5f9"),
                    xaxis=dict(gridcolor="#f1f5f9"),
                )
                st.plotly_chart(fig2, width=True)

        # ── per-query scores ──────────────────────────────────────────────
        st.markdown("**Per-Query Scores**")

        display_df = df.copy()
        if "question" in display_df.columns:
            display_df["question"] = display_df["question"].str[:80] + "…"

        cols_to_show = ["question"] + score_cols if "question" in display_df.columns else score_cols
        st.dataframe(
            display_df[cols_to_show].style.format({c: "{:.3f}" for c in score_cols})
            .background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
            height=400,
        )

        # ── download button ───────────────────────────────────────────────
        st.download_button(
            "⬇️  Download full results CSV",
            data=df.to_csv(index=False),
            file_name="ragas_results.csv",
            mime="text/csv",
        )

    # ── corpus stats ──────────────────────────────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("**Corpus & Pipeline Stats**")

    stats_path = Path("outputs/chunks_stats.json")
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>{stats['total_docs']}</div>"
                f"<div class='metric-label'>Papers</div></div>", unsafe_allow_html=True
            )
        with sc2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>{stats['total_chunks']:,}</div>"
                f"<div class='metric-label'>Chunks</div></div>", unsafe_allow_html=True
            )
        with sc3:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>{int(stats['avg_chunk_tokens'])}</div>"
                f"<div class='metric-label'>Avg Tokens/Chunk</div></div>", unsafe_allow_html=True
            )
        with sc4:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>~7K→484</div>"
                f"<div class='metric-label'>Dedup Pipeline</div></div>", unsafe_allow_html=True
            )
    else:
        st.info("Run the ingest pipeline to populate corpus stats.")