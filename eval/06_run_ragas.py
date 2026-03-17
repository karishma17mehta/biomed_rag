import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset

# Fix for LangChain 1.x / Ragas 0.4.3 compatibility
try:
    import langchain.chains
except ImportError:
    import langchain_classic
    sys.modules["langchain.chains"] = langchain_classic

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Ragas Imports - Using the specific wrappers that solve the 'agenerate_text' error
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance

# -----------------------
# JSONL helpers
# -----------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_contexts(row: Dict[str, Any], top_k: int) -> List[str]:
    hits = (row.get("hits") or [])[:top_k]
    ctxs = []
    for h in hits:
        t = (h.get("text_preview") or h.get("text") or "").strip()
        if t:
            ctxs.append(t)
    return ctxs


import re
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI

def _simple_entities_from_question(q: str) -> List[str]:
    """
    Lightweight heuristic: keep ALL-CAPS-ish gene/drug tokens + key cancer words.
    Replace with your extract_entities() if you want.
    """
    toks = re.findall(r"\b[A-Z0-9\-]{2,}\b", q)
    # keep common genes/drugs (KRAS, EGFR, BRAF, etc.)
    keep = [t for t in toks if any(c.isalpha() for c in t)]
    return list(dict.fromkeys(keep))[:8]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _extract_key_terms(question: str) -> List[str]:
    """
    Heuristic key-term extraction:
    - ALLCAPS gene/drug-ish tokens (KRAS, EGFR, BRAF, MEK, PI3K, etc.)
    - plus cancer words present in question
    Replace with your extract_entities()/extract_cancer_type() if you want.
    """
    q = question or ""
    genes = re.findall(r"\b[A-Z0-9\-]{2,}\b", q)
    genes = [g for g in genes if any(c.isalpha() for c in g)]
    genes = list(dict.fromkeys(genes))[:10]

    cancer_terms = []
    ql = q.lower()
    for t in ["colon", "colorectal", "crc", "rectal", "lung", "nsclc", "pulmonary", "thyroid", "ptc", "ftc", "mtc", "atc"]:
        if t in ql:
            cancer_terms.append(t)

    return genes + cancer_terms

INTENT_TERMS = {
    "response": ["response", "responder", "nonresponder", "efficacy", "benefit", "predict", "predictive"],
    "resistance": ["resistance", "resistant", "refractory", "escape", "acquired", "primary resistance"],
    "survival": ["survival", "overall survival", "os", "pfs", "progression-free", "hazard ratio", "hr"],
    "biomarker": ["biomarker", "marker", "predictive", "prognostic", "wild-type", "mutant", "mutation"],
    "trial": ["phase i", "phase ii", "phase iii", "randomized", "trial", "cohort"],
}

CANCER_MAP = {
    "Colon_Cancer": ["colon", "colorectal", "crc", "rectal", "mcrc"],
    "Lung_Cancer": ["lung", "nsclc", "sclc", "pulmonary"],
    "Thyroid_Cancer": ["thyroid", "ptc", "ftc", "mtc", "atc", "dtc"],
}

CLINICAL_TERMS = ["phase i", "phase ii", "phase iii", "randomized", "trial", "patients", "cohort", "overall survival", "pfs", "os", "response rate", "safety"]
PRECLINICAL_TERMS = ["xenograft", "in vivo", "in vitro", "cell line", "murine", "mouse", "rat", "tumor model"]
AE_TERMS = ["adverse", "adverse event", "ae", "toxicity", "immune-related", "grade", "safety", "pneumonitis", "colitis", "rash", "hepatitis", "endocrine", "nephritis"]

def _infer_intent_terms(question: str) -> list[str]:
    q = (question or "").lower()
    hits = []
    for _, terms in INTENT_TERMS.items():
        for t in terms:
            if t in q:
                hits.extend(terms)
                return list(dict.fromkeys(hits))
    # fallback: keep a small generic set
    return ["response", "resistance", "efficacy", "survival", "predictive", "prognostic"]

def _extract_expected_cancer_terms(question: str) -> list[str]:
    q = (question or "").lower()
    # your question strings often contain "colon cancer"/"lung cancer"/etc
    for k, terms in CANCER_MAP.items():
        if any(t in q for t in terms):
            return terms
    return []

def _extract_entities(q: str, max_n: int = 10) -> List[str]:
    """ALLCAPS-ish gene/drug tokens: KRAS, EGFR, BRAF, PI3K, PD-1, etc."""
    toks = re.findall(r"\b[A-Z0-9\-]{2,}\b", q or "")
    toks = [t for t in toks if any(c.isalpha() for c in t)]
    # de-dupe preserving order
    out = []
    seen = set()
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= max_n:
            break
    return out

def evidence_type(contexts: list[str]) -> str:
    blob = _norm("\n\n".join(contexts))
    clin = sum(1 for t in CLINICAL_TERMS if t in blob)
    pre  = sum(1 for t in PRECLINICAL_TERMS if t in blob)
    if clin >= 2 and clin >= pre:
        return "clinical"
    if pre >= 2 and pre > clin:
        return "preclinical"
    return "unknown"

def _infer_intent_family(question: str) -> str:
    q = _norm(question)
    for fam, terms in INTENT_TERMS.items():
        if any(t in q for t in terms):
            return fam
    return "biomarker"  # safe fallback


def _infer_cancer_terms(question: str) -> List[str]:
    q = _norm(question)
    for _, terms in CANCER_MAP.items():
        if any(t in q for t in terms):
            return terms
    return []  # question didn't specify cancer explicitly


def _needs_clinical(question: str) -> bool:
    q = _norm(question)
    return any(k in q for k in ["clinical", "trial", "phase", "patients", "cohort", "randomized"])


def _needs_ae(question: str) -> bool:
    q = _norm(question)
    return any(k in q for k in ["adverse", "side effect", "toxicity", "safety"])


def _count_term_hits(blob: str, terms: List[str]) -> int:
    """Counts distinct term hits (not raw frequency)."""
    hits = 0
    for t in terms:
        tl = t.lower()
        if tl.isalpha():
            if re.search(rf"\b{re.escape(tl)}\b", blob):
                hits += 1
        else:
            if tl in blob:
                hits += 1
    return hits


def _evidence_type(blob: str) -> str:
    clin = sum(1 for t in CLINICAL_TERMS if t in blob)
    pre  = sum(1 for t in PRECLINICAL_TERMS if t in blob)
    if clin >= 2 and clin >= pre:
        return "clinical"
    if pre >= 2 and pre > clin:
        return "preclinical"
    return "unknown"


def score_context(question: str, ctx: str) -> Tuple[float, Dict]:
    """
    Returns (score, debug_dict).
    Score components (tweak weights as needed):
      - entity overlap (genes/drugs)         max ~ 1.5
      - cancer overlap (if specified)       max ~ 1.0
      - intent overlap (response/survival)  max ~ 1.5
      - clinical match bonus/penalty        +/- 0.7
      - adverse effects match bonus         +0.7 (if AE question)
    """
    blob = _norm(ctx)

    entities = _extract_entities(question)
    cancer_terms = _infer_cancer_terms(question)
    intent_fam = _infer_intent_family(question)
    intent_terms = INTENT_TERMS.get(intent_fam, [])

    ent_hits = _count_term_hits(blob, entities) if entities else 0
    can_hits = _count_term_hits(blob, cancer_terms) if cancer_terms else 0
    int_hits = _count_term_hits(blob, intent_terms)

    # normalize to [0,1] by dividing by how many terms we searched (avoid div0)
    ent_score = (ent_hits / max(1, min(len(entities), 6))) if entities else 0.4  # be mildly permissive
    can_score = (can_hits / max(1, min(len(cancer_terms), 4))) if cancer_terms else 0.5
    int_score = (int_hits / max(1, min(len(intent_terms), 6)))

    # clinical vs preclinical gating
    need_clin = _needs_clinical(question)
    etype = _evidence_type(blob)
    clin_adj = 0.0
    if need_clin:
        if etype == "clinical":
            clin_adj = +0.7
        elif etype == "preclinical":
            clin_adj = -0.7  # strong penalty if user asked "clinical evidence"
        else:
            clin_adj = -0.2  # unknown: small penalty

    # adverse effects gating
    ae_adj = 0.0
    if _needs_ae(question):
        if any(t in blob for t in AE_TERMS):
            ae_adj = +0.7
        else:
            ae_adj = -0.4  # AE question but no AE language

    # Weighted sum
    score = (
        1.5 * ent_score +
        1.0 * can_score +
        1.5 * int_score +
        clin_adj +
        ae_adj
    )

    dbg = {
        "entities": entities[:6],
        "intent_family": intent_fam,
        "need_clinical": need_clin,
        "evidence_type": etype,
        "need_ae": _needs_ae(question),
        "ent_hits": ent_hits,
        "can_hits": can_hits,
        "int_hits": int_hits,
        "ent_score": round(ent_score, 3),
        "can_score": round(can_score, 3),
        "int_score": round(int_score, 3),
        "clin_adj": clin_adj,
        "ae_adj": ae_adj,
        "total": round(score, 3),
    }
    return score, dbg


def filter_contexts(
    question: str,
    contexts: List[str],
    keep_k: int = 6,
    min_score: float = 1.6,
    fallback_k: int = 4,
    debug: bool = False,
) -> List[str]:
    """
    Intent-aware context filtering:
      1) score each context
      2) keep the top `keep_k` above `min_score`
      3) if none pass threshold, fall back to top `fallback_k` (never return empty)
    """
    if not contexts:
        return []

    scored = []
    for c in contexts:
        s, info = score_context(question, c)
        scored.append((s, c, info))

    scored.sort(key=lambda x: x[0], reverse=True)

    kept = [c for (s, c, _) in scored if s >= min_score][:keep_k]
    if not kept:
        kept = [c for (_, c, _) in scored[:fallback_k]]

    if debug:
        print("\n--- CONTEXT SCORING DEBUG ---")
        for i, (s, _, info) in enumerate(scored[:10], 1):
            print(f"{i:02d}. score={s:.3f}  info={info}")

    return kept



def _evidence_pass(question: str, contexts: list[str]) -> tuple[bool, dict]:
    """
    Require at least 2 of these 3 signals:
      1) entity signal (KRAS/BRAF/EGFR etc.)
      2) cancer signal (if question specifies a cancer)
      3) intent signal (response/resistance/survival/etc.)
    """
    q_terms_entities = _simple_entities_from_question(question)
    q_terms_cancer = _extract_expected_cancer_terms(question)
    q_terms_intent = _infer_intent_terms(question)

    blob = _norm("\n\n".join(contexts))

    def any_hit(terms: list[str]) -> bool:
        for t in terms:
            tl = t.lower()
            if tl.isalpha():
                if re.search(rf"\b{re.escape(tl)}\b", blob):
                    return True
            else:
                if tl in blob:
                    return True
        return False

    entity_ok = any_hit(q_terms_entities) if q_terms_entities else True
    cancer_ok = any_hit(q_terms_cancer) if q_terms_cancer else True  # only required if specified
    intent_ok = any_hit(q_terms_intent)

    # if cancer is specified, count it; if not, don’t penalize
    signals = 0
    signals += 1 if entity_ok else 0
    signals += 1 if (cancer_ok if q_terms_cancer else True) else 0
    signals += 1 if intent_ok else 0

    has_evidence = signals >= 2

    dbg = {
        "entities": q_terms_entities,
        "cancer_terms": q_terms_cancer,
        "intent_terms": q_terms_intent[:10],
        "entity_ok": entity_ok,
        "cancer_ok": cancer_ok,
        "intent_ok": intent_ok,
        "signals": signals,
    }
    return has_evidence, dbg



def generate_grounded_answer(
    chat_llm: ChatOpenAI,
    question: str,
    contexts: List[str],
    max_ctx: int = 6,
    debug: bool = False,
) -> str:
    ctxs = (contexts or [])[:max_ctx]
    if not ctxs:
        return "Insufficient context: no retrieved passages were provided."

    # ---------- Special-case: adverse effects ----------
    # If the question is about adverse effects/toxicity, require AE signal.
    if question_requires_ae(question) and not has_ae_signal(ctxs):
        joined = "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(ctxs)])
        # keep this SHORT to avoid relevance penalties from fluff
        prompt = f"""You are a biomedical assistant.

Use ONLY the contexts.
The question asks about adverse effects/toxicity, but the contexts may not include safety/adverse-event reporting.

Write ONE sentence:
- If adverse effects are not described, say that explicitly.
- End with citations.

QUESTION:
{question}

CONTEXTS:
{joined}

One-sentence answer:"""
        resp = chat_llm.invoke(prompt)
        return (resp.content or "").strip()

    # ---------- Evidence detection (your existing pass) ----------
    has_evidence, ev = _evidence_pass(question, ctxs)
    if debug:
        print("EVIDENCE_PASS:", has_evidence, ev)

    if not has_evidence:
        missing_bits = []
        if not ev.get("entity_ok", True):
            missing_bits.append("the key entity/entities from the question")
        if ev.get("cancer_terms") and not ev.get("cancer_ok", True):
            missing_bits.append("the expected cancer context")
        if not ev.get("intent_ok", True):
            missing_bits.append("the key outcome/intent (e.g., response/resistance/survival)")
        miss = "; ".join(missing_bits) if missing_bits else "the key information needed"
        return f"Insufficient context: the retrieved passages do not contain {miss}."

    # ---------- Special-case: clinical vs preclinical mismatch ----------
    ql = (question or "").lower()
    wants_clinical = any(k in ql for k in ["clinical", "phase", "trial", "patients", "cohort", "randomized"])
    etype = evidence_type(ctxs)

    if wants_clinical and etype == "preclinical":
        joined = "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(ctxs)])
        prompt = f"""You are a biomedical assistant.

RULES:
- Use ONLY the provided contexts.
- The question asks for CLINICAL evidence, but the contexts appear PRECLINICAL (cell lines/xenografts/in vitro/in vivo).
- Write 2 sentences:
  1) State that only preclinical evidence is present and summarize what it shows.
  2) State that clinical trial outcomes are not provided in these contexts.
- Every sentence must end with citations.

QUESTION:
{question}

CONTEXTS:
{joined}

Write the 2-sentence answer:"""
        resp = chat_llm.invoke(prompt)
        return (resp.content or "").strip()

    # ---------- Default grounded answer ----------
    joined = "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(ctxs)])

    prompt = f"""You are a biomedical assistant answering with retrieval.

HARD RULES:
- Use ONLY the provided contexts. No outside knowledge.
- Output 3 to 5 sentences (not fixed at 4).
- Each sentence MUST end with citations like [CTX 1] or [CTX 2][CTX 4].
- Answer the question directly first, then support with evidence, then state limitations only if needed.

QUESTION:
{question}

CONTEXTS:
{joined}

Write the answer now:"""

    resp = chat_llm.invoke(prompt)
    return (resp.content or "").strip()


# -----------------------
# Metric Loading
# -----------------------


def get_metrics(llm: LangchainLLMWrapper, embeddings: LangchainEmbeddingsWrapper):
    """
    Returns initialized Ragas metric objects.
    """
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRelevance(llm=llm)
    ]

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="eval/runs/retrieval_run_ragas.jsonl")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--out", default="eval/runs/ragas_results.csv")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.run))

    # 1) Standard LangChain objects
    # We use these for generation and as the base for Ragas wrappers
    chat_llm = ChatOpenAI(model=args.model, temperature=0)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # 2) Ragas "Modern" Wrappers
    # This manually bridges LangChain to Ragas 0.4.3+ and provides 'agenerate_text'
    ragas_llm = LangchainLLMWrapper(chat_llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings_model)

    # 3) Build Dataset
    print(f"Building dataset for {len(rows)} queries...")
    data = []
    for r in rows:
        question = (r.get("query") or r.get("question") or "").strip()
        if not question:
            continue
        contexts = build_contexts(r, args.top_k)
        answer = generate_grounded_answer(chat_llm, question, contexts) if contexts else "Insufficient context."
        data.append({
            "question": question, 
            "answer": answer, 
            "contexts": contexts
        })

    dataset = Dataset.from_pandas(pd.DataFrame(data))

    # 4) Initialize Metrics
    metrics = get_metrics(llm=ragas_llm, embeddings=ragas_emb)

    # 5) Evaluate
    # Passing the wrapped objects directly ensures compliance with Ragas' internal validation
    print("Starting Ragas evaluation...")
    result = evaluate(
        dataset=dataset, 
        metrics=metrics, 
        llm=ragas_llm, 
        embeddings=ragas_emb
    )

    # 6) Save Results
    df = result.to_pandas()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n=== RAGAS SUMMARY ===")
    print(result)
    print(f"\nSaved per-row scores to: {out_path}")

if __name__ == "__main__":
    main()
