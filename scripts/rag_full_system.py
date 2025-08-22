# scripts/rag_full_system.py
"""
Complete RAG full system implementing:
- Hybrid retrieval (FAISS dense + BM25 sparse)
- Reciprocal Rank Fusion (RRF) or weighted fusion
- Multi-stage retrieval with CrossEncoder reranking
- Response generation using a small LM (DistilGPT2 default)
- Input and output guardrails
"""

import os
import re
import time
import json
import pickle
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------
# Paths — adjust if your layout differs
# ---------------------------
def _project_paths() -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    return {
        "root": project_root,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "cross_encoder": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "generator": "distilgpt2",
        "faiss": os.path.join(project_root, "data", "faiss_index.bin"),
        "chunks": os.path.join(project_root, "data", "rag_chunks.json"),
        "bm25_tokens": os.path.join(project_root, "data", "bm25_corpus.json"),
        "bm25_pickle": os.path.join(project_root, "data", "bm25_obj.pkl"),
    }

P = _project_paths()

# ---------------------------
# Stopwords & finance keywords
# ---------------------------
STOPWORDS = set("""
a an and are as at be by for from has have in into is it its of on or s shall should such that the their there these they this to was were will with within without would about above after again against all am any because been before being below between both can did do does doing down during each few further he her here hers herself him himself his how i if into itself just ll me more most my myself no nor not now off once only other our ours ourselves out over own same she so some than then there s theirs them themselves these those through under until up very we what when where which who whom why you your yours yourself yourselves
""".split())

FINANCE_KEYWORDS = [
    "revenue","net sales","sales","gross margin","gross profit","operating income",
    "operating expenses","net income","eps","earnings per share","assets","liabilities",
    "cash","cash flows","cashflow","free cash flow","dividends","share repurchase",
    "cost of sales","balance sheet","income statement","cash flows","operating cash",
    "investing cash","financing cash","retained earnings","deferred revenue"
]

# ---------------------------
# Utilities
# ---------------------------
def preprocess_query(q: str) -> Tuple[str, List[str]]:
    """Clean, lowercase, tokenize for BM25 (also used for simple checks)."""
    q_clean = re.sub(r"\s+", " ", q).strip().lower()
    tokens = [w for w in re.split(r"[^a-z0-9$%.-]+", q_clean) if w and w not in STOPWORDS]
    return q_clean, tokens

def extract_years(q: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]

# ---------------------------
# Guardrails
# ---------------------------
def input_guardrail(q: str) -> Tuple[bool, str]:
    q_clean, tokens = preprocess_query(q)
    if len(q_clean) < 8 or len(tokens) < 2:
        return False, "Query too short/ambiguous."
    if not any(k in q_clean for k in FINANCE_KEYWORDS):
        return False, "Query appears unrelated to financial statements."
    return True, "OK"

def output_guardrail(answer: str, passages: List[str], query: str) -> Tuple[bool, str]:
    """Numeric grounding + metric & year checks."""
    if not answer or answer.startswith("__"):
        return False, "No valid answer produced."

    ctx = " ".join(passages).lower().replace(",", "")
    ans = answer.lower().replace(",", "")

    # numeric grounding
    nums = re.findall(r"\$?[0-9]+(?:\.[0-9]+)?%?", ans)
    grounded = True
    for n in nums:
        if n.endswith("%"):
            if n not in ctx and n[:-1] not in ctx:
                grounded = False; break
        else:
            n0 = n.replace("$", "")
            if n0 not in ctx:
                grounded = False; break

    metric_present = any(k in ans for k in [m for m, _ in FIN_METRIC_PATTERNS])
    years = extract_years(query)
    year_ok = True
    if years:
        year_ok = any(str(y) in ctx for y in years)

    ok = grounded and metric_present and year_ok
    if ok:
        return True, "Answer appears grounded."
    reasons = []
    if not grounded: reasons.append("numbers not grounded")
    if not metric_present: reasons.append("metric missing")
    if not year_ok: reasons.append("year mismatch")
    return False, "Warning: " + ", ".join(reasons)

# ---------------------------
# Loading components
# ---------------------------
def load_all_components(use_gpu_if_available: bool = True, fallback_generator: bool = True):
    logging.info("Loading models & indexes...")
    device = "cuda" if (torch.cuda.is_available() and use_gpu_if_available) else "cpu"

    # Embedding model
    embedding_model = SentenceTransformer(P["embed_model"], device=device)
    logging.info("Embedding model loaded.")

    # Cross-encoder (reranker)
    cross_encoder_model = CrossEncoder(P["cross_encoder"], device=device)
    logging.info("Cross-encoder loaded.")

    # FAISS index
    if not os.path.exists(P["faiss"]):
        raise FileNotFoundError(f"FAISS index missing at {P['faiss']}")
    faiss_index = faiss.read_index(P["faiss"])
    logging.info("FAISS index loaded.")

    # Chunks JSON
    if not os.path.exists(P["chunks"]):
        raise FileNotFoundError(f"Chunks JSON missing at {P['chunks']}")
    with open(P["chunks"], "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    # Normalize to texts + metadatas arrays (aligned with FAISS index order)
    texts = []
    metadatas = []
    if isinstance(chunks_data, dict) and 'contents' in chunks_data and 'metadatas' in chunks_data:
        texts = chunks_data['contents']
        metadatas = chunks_data['metadatas']
    else:
        for item in chunks_data:
            if isinstance(item, dict):
                texts.append(item.get('page_content') or item.get('text') or "")
                metadatas.append(item.get('metadata') or {"source": item.get("heading") or None, "id": item.get("id")})
            else:
                texts.append(str(item))
                metadatas.append({})
    logging.info(f"Loaded {len(texts)} chunks.")

    # BM25: load tokenized corpus (list of token lists aligned with texts)
    bm25_index = None
    if os.path.exists(P["bm25_pickle"]):
        try:
            with open(P["bm25_pickle"], "rb") as f:
                bm25_index = pickle.load(f)
            logging.info("BM25 object loaded from pickle.")
        except Exception:
            bm25_index = None

    if bm25_index is None:
        if not os.path.exists(P["bm25_tokens"]):
            raise FileNotFoundError(f"BM25 tokens missing at {P['bm25_tokens']}")
        with open(P["bm25_tokens"], "r", encoding="utf-8") as f:
            tokenized_corpus = json.load(f)
        bm25_index = BM25Okapi(tokenized_corpus)
        try:
            with open(P["bm25_pickle"], "wb") as f:
                pickle.dump(bm25_index, f)
        except Exception:
            logging.warning("Could not pickle BM25 object; continuing without pickle.")

    # Local generator (fallback)
    generator = None
    if fallback_generator:
        try:
            gen_tokenizer = AutoTokenizer.from_pretrained(P["generator"])
            gen_model = AutoModelForCausalLM.from_pretrained(P["generator"])
            gen_model.to(device)
            generator = (gen_tokenizer, gen_model, device)
            logging.info("Local generator (DistilGPT2) loaded.")
        except Exception as e:
            logging.warning(f"Local generator load failed: {e}. It will be disabled.")

    logging.info("All components loaded.")
    return embedding_model, faiss_index, texts, metadatas, bm25_index, cross_encoder_model, generator

# ---------------------------
# Dense retrieval (FAISS)
# ---------------------------
def get_dense_candidates(query: str, faiss_index, embedding_model, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    q_emb = embedding_model.encode(query, convert_to_numpy=True)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1).astype("float32")
    else:
        q_emb = q_emb.astype("float32")
    scores, indices = faiss_index.search(q_emb, top_k)
    return scores.flatten(), indices.flatten()

# ---------------------------
# Sparse retrieval (BM25)
# ---------------------------
def get_bm25_candidates(query_tokens: List[str], bm25_index: BM25Okapi, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    scores = bm25_index.get_scores(query_tokens)
    idxs = np.argsort(scores)[::-1][:top_k]
    return scores[idxs], idxs

# ---------------------------
# Fusion functions
# ---------------------------
def reciprocal_rank_fusion(rank_lists: Dict[str, List[int]], k: int = 60, topn: int = 50) -> List[int]:
    scores = {}
    for _, ranks in rank_lists.items():
        for r, idx in enumerate(ranks):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + r + 1.0)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in fused[:topn]]

def weighted_score_fusion(dense_scores: np.ndarray, dense_idx: np.ndarray,
                          sparse_scores: np.ndarray, sparse_idx: np.ndarray,
                          dense_weight: float = 0.6, sparse_weight: float = 0.4,
                          topn: int = 50) -> List[int]:
    score_map = {}
    dense_sim = -dense_scores  # convert L2 distance to similarity
    for idx, sim in zip(dense_idx, dense_sim):
        score_map[int(idx)] = score_map.get(int(idx), 0.0) + dense_weight * float(sim)
    for idx, s in zip(sparse_idx, sparse_scores):
        score_map[int(idx)] = score_map.get(int(idx), 0.0) + sparse_weight * float(s)
    fused = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in fused[:topn]]

# ---------------------------
# Hybrid retrieval
# ---------------------------
def hybrid_retrieve(query: str,
                    faiss_index,
                    faiss_chunks: List[str],
                    bm25_index: BM25Okapi,
                    embedding_model,
                    candidate_k: int = 50,
                    fusion: str = "rrf") -> List[int]:
    q_clean, tokens = preprocess_query(query)
    dense_scores, dense_idx = get_dense_candidates(q_clean, faiss_index, embedding_model, top_k=candidate_k)
    sparse_scores, sparse_idx = get_bm25_candidates(tokens, bm25_index, top_k=candidate_k)
    dense_list = dense_idx.tolist()
    sparse_list = sparse_idx.tolist()
    if fusion == "rrf":
        fused = reciprocal_rank_fusion({"dense": dense_list, "sparse": sparse_list}, k=60, topn=candidate_k)
    elif fusion == "weighted":
        fused = weighted_score_fusion(dense_scores, dense_idx, sparse_scores, sparse_idx,
                                      dense_weight=0.6, sparse_weight=0.4, topn=candidate_k)
    else:
        seen = set(); fused = []
        for idx in dense_list + sparse_list:
            if int(idx) not in seen:
                seen.add(int(idx)); fused.append(int(idx))
        fused = fused[:candidate_k]
    return fused

# ---------------------------
# Cross-encoder rerank
# ---------------------------
def rerank_with_cross_encoder(query: str,
                              candidate_indices: List[int],
                              faiss_chunks: List[str],
                              cross_encoder_model,
                              top_k: int = 5,
                              batch_size: int = 32) -> Tuple[List[str], List[float], List[int]]:
    pairs = [(query, faiss_chunks[i]) for i in candidate_indices]
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        s = cross_encoder_model.predict(batch)
        scores.extend(s.tolist())
    scores = np.array(scores)
    order = np.argsort(scores)[::-1][:top_k]
    top_passages = [faiss_chunks[candidate_indices[i]] for i in order]
    top_scores = [float(scores[i]) for i in order]
    top_indices = [candidate_indices[i] for i in order]
    return top_passages, top_scores, top_indices

# ---------------------------
# Deterministic extractor (regex)
# ---------------------------
FIN_METRIC_PATTERNS = [
    ("net sales", r"(?:total\s+)?net\s+sales[:\s]*\$?\s*([0-9][0-9,\.]*)"),
    ("revenue", r"revenue[:\s]*\$?\s*([0-9][0-9,\.]*)"),
    ("gross margin", r"gross\s+margin(?:\s+percentage|\s*\(%)?\s*[:\s]*([0-9]{1,3}\.?[0-9]?)\%|gross\s+margin[:\s]*\$?\s*([0-9][0-9,\.]*)"),
    ("operating income", r"operating\s+income[:\s]*\$?\s*([0-9][0-9,\.]*)"),
    ("net income", r"net\s+income[:\s]*\$?\s*([0-9][0-9,\.]*)"),
    ("dividends", r"dividends\s+paid[:\s]*\$?\s*([0-9][0-9,\.]*)"),
]

def _pick_first_group(m):
    if not m: return None
    for g in m.groups():
        if g: return g
    return None

def regex_extract_answer(query: str, passages: List[str]) -> str:
    ctx = "\n---\n".join(passages)
    ql = query.lower()
    years = extract_years(ql)
    for metric, pattern in FIN_METRIC_PATTERNS:
        if metric in ql:
            m = re.search(pattern, ctx, flags=re.IGNORECASE)
            val = _pick_first_group(m)
            if val:
                val = val.strip()
                ytxt = f" in {years[0]}" if years else ""
                if not val.endswith("%") and not val.startswith("$"):
                    val = f"${val}"
                return f"Apple’s {metric}{ytxt} was {val}."
    for metric, pattern in FIN_METRIC_PATTERNS:
        m = re.search(pattern, ctx, flags=re.IGNORECASE)
        val = _pick_first_group(m)
        if val:
            val = val.strip()
            if not val.endswith("%") and not val.startswith("$"):
                val = f"${val}"
            ytxt = f" in {years[0]}" if years else ""
            return f"Apple’s {metric}{ytxt} was {val}."
    return "__FALLBACK__"

# ---------------------------
# Local generator fallback
# ---------------------------
def local_generate_fallback(query: str, passages: List[str], generator, max_tokens_out: int = 64) -> str:
    if not generator:
        return "Answer not found in the provided financial data."
    tokenizer, model, device = generator
    context = "\n\n".join(passages)
    try:
        max_ctx = tokenizer.model_max_length or getattr(model.config, "n_positions", 1024)
    except Exception:
        max_ctx = 1024
    max_chars = int(max_ctx * 3)
    if len(context) > max_chars:
        context = context[:max_chars]
    prompt = (
        "Answer concisely using ONLY the context. If unknown, say 'Not found.'\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens_out, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    ans = text.split("Answer:")[-1].strip()
    return ans[:512] if ans else "Answer not found in the provided financial data."

# ---------------------------
# Orchestration: multi-stage retrieval + generation (full answer)
# ---------------------------
def multi_stage_retrieval(query: str,
                          faiss_index,
                          faiss_chunks,
                          bm25_index,
                          embedding_model,
                          cross_encoder_model,
                          top_k: int = 5,
                          candidate_k: int = 50,
                          fusion_method: str = "rrf"):
    logging.info("Stage 1: Hybrid retrieval")
    candidates = hybrid_retrieve(query=query,
                                 faiss_index=faiss_index,
                                 faiss_chunks=faiss_chunks,
                                 bm25_index=bm25_index,
                                 embedding_model=embedding_model,
                                 candidate_k=candidate_k,
                                 fusion=fusion_method)
    logging.info(f"Candidates retrieved: {len(candidates)}. Stage 2: Cross-encoder reranking ...")
    passages, rerank_scores, top_indices = rerank_with_cross_encoder(query=query,
                                                                     candidate_indices=candidates,
                                                                     faiss_chunks=faiss_chunks,
                                                                     cross_encoder_model=cross_encoder_model,
                                                                     top_k=top_k)
    return passages, rerank_scores, top_indices

def run_rag_system(query: str, components, top_k: int = 5, candidate_k: int = 60, fusion: str = "rrf"):
    """Full pipeline: guardrail -> retrieve -> rerank -> extract/generate -> output guardrail -> return structured result"""
    embedding_model, faiss_index, faiss_chunks, faiss_metadatas, bm25_index, cross_encoder_model, generator = components

    # Input guardrail
    ok, msg = input_guardrail(query)
    if not ok:
        return {
            "answer": "Query out of scope.",
            "retrieval_confidence": 0.0,
            "response_time": 0.0,
            "is_grounded": False,
            "guardrail_message": msg,
            "retrieved_passages": [],
            "retrieved_metadata": []
        }

    t0 = time.time()

    # Multi-stage retrieval (hybrid + rerank)
    passages, rerank_scores, top_indices = multi_stage_retrieval(query=query,
                                                                 faiss_index=faiss_index,
                                                                 faiss_chunks=faiss_chunks,
                                                                 bm25_index=bm25_index,
                                                                 embedding_model=embedding_model,
                                                                 cross_encoder_model=cross_encoder_model,
                                                                 top_k=top_k,
                                                                 candidate_k=candidate_k,
                                                                 fusion_method=fusion)

    metadatas = [faiss_metadatas[i] if i < len(faiss_metadatas) else {} for i in top_indices]

    # Deterministic extractor first
    answer = regex_extract_answer(query, passages)

    # Fallback generator if extraction failed
    if answer == "__FALLBACK__":
        answer = local_generate_fallback(query, passages, generator)

    # Output guardrail
    is_grounded, guardrail_message = output_guardrail(answer, passages, query)

    t1 = time.time()
    response_time = t1 - t0
    retrieval_confidence = float(rerank_scores[0]) if rerank_scores else 0.0

    return {
        "answer": answer,
        "retrieval_confidence": retrieval_confidence,
        "response_time": response_time,
        "is_grounded": is_grounded,
        "guardrail_message": guardrail_message,
        "retrieved_passages": passages,
        "retrieved_metadata": metadatas
    }

# ---------------------------
# CLI demo
# ---------------------------
if __name__ == "__main__":
    components = load_all_components()
    print("\nRAG system ready. Try sample queries.\n")

    test_queries = [
        "What were Apple's total net sales in 2024?",
        "What was Apple's gross margin percentage in 2023?",
        "How much dividends were paid in 2024?",
        "Tell me the capital of France."  # should be rejected by guardrail
    ]

    for q in test_queries:
        print(f"\n=== Query: {q}")
        res = run_rag_system(q, components, top_k=3, candidate_k=80, fusion="rrf")
        print("Answer:", res["answer"])
        print("Grounded:", res["is_grounded"], "| Guardrail:", res["guardrail_message"])
        print(f"Response time: {res['response_time']:.2f}s | Retrieval confidence (top rerank score): {res['retrieval_confidence']:.4f}")
        print("\nTop passages (truncated) & metadata:")
        for p, m in zip(res["retrieved_passages"], res["retrieved_metadata"]):
            src = m.get("source", "unknown")
            print(f"- [src: {src}] {p[:200].replace('\\n', ' ')} ...")
