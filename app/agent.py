# app/agent.py
"""
Biomedical RAG Agent — LangGraph implementation
Two tools:
  1. search_local_index  — queries your FAISS/BM25 hybrid index
  2. search_pubmed_live  — falls back to NCBI Entrez API (no key needed)

The agent decides which tool(s) to call based on local result quality,
then generates a grounded answer from the combined context.
"""

import os
import re
import time
import ssl
import json
import urllib.request
import urllib.parse
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# ── Local retrieval ────────────────────────────────────────────────────────────
from app.retrieve_faiss import retrieve
from app.query_router import extract_entities, extract_cancer_type

# ── Constants ─────────────────────────────────────────────────────────────────
LOCAL_SCORE_THRESHOLD = 0.55   # min avg score to trust local results
LOCAL_MIN_HITS        = 2      # need at least this many local hits above threshold
PUBMED_MAX_RESULTS    = 6      # articles to fetch from PubMed
PUBMED_ABSTRACT_CHARS = 1200   # truncate each abstract to this length

CANCER_MESH = {
    "Colon_Cancer":   "colorectal neoplasms",
    "Lung_Cancer":    "lung neoplasms",
    "Thyroid_Cancer": "thyroid neoplasms",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Local FAISS index
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_local_index(query: str) -> str:
    """
    Search the local biomedical FAISS index for relevant chunks.
    Returns top chunks with scores. Use this first for any biomedical question.
    """
    try:
        hits = retrieve(query, top_n_dense=200, top_n_bm25=200, top_k=8,
                        max_per_paper=1, max_per_hash=1)
        if not hits:
            return json.dumps({"status": "no_results", "chunks": []})

        chunks = []
        for h in hits:
            text = h.get("text", "")
            # strip metadata header (everything before ---)
            parts = re.split(r"\s*---\s*", text, maxsplit=1)
            body = parts[1].strip() if len(parts) == 2 else text.strip()
            chunks.append({
                "score":    round(float(h.get("score", 0)), 4),
                "paper_id": h.get("paper_id", ""),
                "section":  h.get("section", ""),
                "text":     body[:1000],
            })

        avg_score = sum(c["score"] for c in chunks) / max(1, len(chunks))
        strong    = [c for c in chunks if c["score"] >= LOCAL_SCORE_THRESHOLD]

        return json.dumps({
            "status":    "ok",
            "avg_score": round(avg_score, 4),
            "n_strong":  len(strong),
            "chunks":    chunks,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "chunks": []})


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: PubMed live search via NCBI Entrez (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────
def _pubmed_search_ids(query: str, max_results: int = 6) -> List[str]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = urllib.parse.urlencode({
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    })
    url = f"{base}?{params}"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=10, context=ctx) as r:
            data = json.loads(r.read().decode())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def _pubmed_fetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    if not pmids:
        return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = urllib.parse.urlencode({
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    })
    url = f"{base}?{params}"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=15, context=ctx) as r:
            xml = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"PubMed fetch error: {e}")
        return []

    articles = []
    for article_xml in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL):
        pmid_m     = re.search(r"<PMID[^>]*>(\d+)</PMID>", article_xml)
        title_m    = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article_xml, re.DOTALL)
        abstract_m = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", article_xml, re.DOTALL)
        year_m     = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", article_xml, re.DOTALL)

        pmid     = pmid_m.group(1)                                          if pmid_m     else "?"
        title    = re.sub(r"<[^>]+>", "", title_m.group(1))                if title_m    else "No title"
        abstract = re.sub(r"<[^>]+>", "", abstract_m.group(1))             if abstract_m else "No abstract"
        year     = year_m.group(1)                                          if year_m     else "?"

        articles.append({
            "pmid":     pmid,
            "title":    title.strip(),
            "abstract": abstract.strip()[:PUBMED_ABSTRACT_CHARS],
            "year":     year,
            "source":   f"PubMed PMID:{pmid} ({year})",
        })
    return articles


def _build_pubmed_query(query: str) -> str:
    entities    = extract_entities(query)
    cancer_type = extract_cancer_type(query)
    mesh        = CANCER_MESH.get(cancer_type, "")

    # Use simple terms without field tags — more permissive
    ent_terms   = " AND ".join(entities[:2]) if entities else ""
    cancer_term = mesh if mesh else ""

    parts = [p for p in [ent_terms, cancer_term] if p]
    if parts:
        return " AND ".join(parts)

    # fallback: key words from query
    words = [w for w in re.findall(r"[A-Za-z]{3,}", query) if w.lower() not in
             {"what","which","how","does","are","the","for","with","that","this","from"}]
    return " ".join(words[:6])

@tool
def search_pubmed_live(query: str) -> str:
    """
    Search PubMed live for recent biomedical abstracts.
    Use this when local index results are weak or corpus gaps are suspected
    (especially for colorectal BRAF, thyroid cell lines, immunotherapy AEs).
    """
    try:
        pubmed_query = _build_pubmed_query(query)
        pmids = _pubmed_search_ids(pubmed_query, max_results=PUBMED_MAX_RESULTS)

        # small delay to respect NCBI rate limit (3 req/sec without API key)
        time.sleep(0.4)

        if not pmids:
            # fallback: simpler query
            simple = " ".join(query.split()[:6])
            pmids  = _pubmed_search_ids(simple, max_results=PUBMED_MAX_RESULTS)

        articles = _pubmed_fetch_abstracts(pmids)

        if not articles:
            return json.dumps({"status": "no_results", "articles": []})

        return json.dumps({
            "status":       "ok",
            "pubmed_query": pubmed_query,
            "n_results":    len(articles),
            "articles":     articles,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "articles": []})


# ─────────────────────────────────────────────────────────────────────────────
# Agent state
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:        str
    messages:        Annotated[List, lambda x, y: x + y]
    local_chunks:    List[Dict]
    pubmed_articles: List[Dict]
    local_weak:      bool
    final_answer:    str


# ─────────────────────────────────────────────────────────────────────────────
# Node: call local index
# ─────────────────────────────────────────────────────────────────────────────
def node_local_search(state: AgentState) -> AgentState:
    question = state["question"]
    result_json = search_local_index.invoke({"query": question})
    result = json.loads(result_json)

    chunks   = result.get("chunks", [])
    n_strong = result.get("n_strong", 0)

    entities    = extract_entities(question)
    cancer_type = extract_cancer_type(question)
    primary_entities = [e for e in entities[:5] if len(e) >= 3] if entities else []

    entity_present = False
    if primary_entities:
        all_text = " ".join(c["text"].lower() for c in chunks)
        entity_present = any(e.lower() in all_text for e in primary_entities)
    else:
        entity_present = True

    # ✅ NEW: if cancer is specified, also check that entity+cancer co-occur
    # in at least one chunk (not just entity anywhere + cancer anywhere)
    cooccurrence_ok = True
    if primary_entities and cancer_type:
        cancer_keywords = {
            "Thyroid_Cancer": ["thyroid"],
            "Colon_Cancer":   ["colon", "colorectal", "crc"],
            "Lung_Cancer":    ["lung", "nsclc"],
        }.get(cancer_type, [])
        cooccurrence_ok = any(
            any(re.search(rf'\b{re.escape(e.lower())}\b', c["text"].lower()) for e in primary_entities)
            and any(k in c["text"].lower() for k in cancer_keywords)
            for c in chunks
        )

    local_weak = (n_strong < LOCAL_MIN_HITS) or (primary_entities and not entity_present) or not cooccurrence_ok

    return {
        "local_chunks": chunks,
        "local_weak":   local_weak,
        "messages": [AIMessage(content=f"Local search: {len(chunks)} chunks, {n_strong} strong, entity_present={entity_present}, cooccurrence_ok={cooccurrence_ok}. Weak={local_weak}")],
    }

# ─────────────────────────────────────────────────────────────────────────────
# Node: call PubMed (only when local is weak)
# ─────────────────────────────────────────────────────────────────────────────
def node_pubmed_search(state: AgentState) -> AgentState:
    question = state["question"]
    result_json = search_pubmed_live.invoke({"query": question})
    result = json.loads(result_json)

    articles = result.get("articles", [])
    return {
        "pubmed_articles": articles,
        "messages": [AIMessage(content=f"PubMed search: {len(articles)} abstracts retrieved.")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: generate final answer
# ─────────────────────────────────────────────────────────────────────────────
def node_generate(state: AgentState) -> AgentState:
    question        = state["question"]
    local_chunks    = state.get("local_chunks", [])
    pubmed_articles = state.get("pubmed_articles", [])

    # Build context block
    ctx_parts = []
    for i, c in enumerate(local_chunks[:5], 1):
        ctx_parts.append(f"[LOCAL {i}] (paper: {c.get('paper_id','?')}, section: {c.get('section','?')})\n{c['text']}")

    for i, a in enumerate(pubmed_articles[:4], 1):
        ctx_parts.append(f"[PUBMED {i}] {a['title']} ({a['year']})\n{a['abstract']}")

    if not ctx_parts:
        return {"final_answer": "Insufficient context: no relevant passages found in local index or PubMed."}

    ctx_block = "\n\n".join(ctx_parts)
    has_pubmed = len(pubmed_articles) > 0

    source_note = (
        "Sources include both your local corpus and live PubMed abstracts."
        if has_pubmed else
        "Sources are from your local biomedical corpus."
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    prompt = f"""You are a biomedical research assistant. {source_note}

STRICT RULES:
1. Use ONLY information from the contexts below. No outside knowledge.
2. Every sentence MUST end with a citation like [LOCAL 1] or [PUBMED 2].
3. If contexts lack sufficient information, say: "Insufficient evidence in provided contexts."
4. Write 3 to 5 sentences. Answer the question directly in the first sentence.
5. Do NOT speculate or use phrases like "it is known that" without a citation.

QUESTION: {question}

CONTEXTS:
{ctx_block}

Grounded answer (every sentence must have a citation):"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=f"Answer generated. Used {len(local_chunks)} local + {len(pubmed_articles)} PubMed sources.")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing: should we call PubMed?
# ─────────────────────────────────────────────────────────────────────────────
def route_after_local(state: AgentState) -> str:
    return "pubmed" if state.get("local_weak", False) else "generate"


# ─────────────────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)

    g.add_node("local_search", node_local_search)
    g.add_node("pubmed_search", node_pubmed_search)
    g.add_node("generate",     node_generate)

    g.set_entry_point("local_search")
    g.add_conditional_edges("local_search", route_after_local, {
        "pubmed":   "pubmed_search",
        "generate": "generate",
    })
    g.add_edge("pubmed_search", "generate")
    g.add_edge("generate", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────
_agent = None

def get_agent():
    return build_agent()


def run_agent(question: str) -> Dict[str, Any]:
    """
    Main entry point. Returns dict with:
      - answer: str
      - local_chunks: list
      - pubmed_articles: list
      - used_pubmed: bool
    """
    agent = get_agent()
    result = agent.invoke({
        "question":        question,
        "messages":        [HumanMessage(content=question)],
        "local_chunks":    [],
        "pubmed_articles": [],
        "local_weak":      False,
        "final_answer":    "",
    })
    return {
        "answer":          result["final_answer"],
        "local_chunks":    result.get("local_chunks", []),
        "pubmed_articles": result.get("pubmed_articles", []),
        "used_pubmed":     len(result.get("pubmed_articles", [])) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the role of BRAF mutations in papillary thyroid carcinoma?"
    print(f"\nQ: {q}\n")

    result = run_agent(q)
    print(f"Used PubMed: {result['used_pubmed']}")
    print(f"Local chunks: {len(result['local_chunks'])}")
    print(f"PubMed articles: {len(result['pubmed_articles'])}")
    print(f"\nANSWER:\n{result['answer']}")