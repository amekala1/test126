# scripts/preprocessing.py

import os
import re
import json
import uuid
import logging
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
import faiss
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Optional OCR fallback (only used if import succeeds and page text is empty)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------

def safe_write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def norm_space(s: str) -> str:
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def simple_clean(text: str) -> str:
    # Remove repeated headers/footers & noise patterns commonly seen in 10-Ks
    text = re.sub(r"Apple Inc\. \| \d{4} Form 10-K", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\f\s+", "\n", text)  # form feed
    text = norm_space(text)
    return text

SECTION_PATTERN = re.compile(
    r"(ITEM\s+\d+\.[^\n]+|PART\s+[I-V]+\b|FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA|"
    r"CONSOLIDATED STATEMENTS OF [A-Z ][A-Z ]+|CONSOLIDATED BALANCE SHEETS|"
    r"CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME|"
    r"CONSOLIDATED STATEMENTS OF CASH FLOWS|"
    r"CONSOLIDATED STATEMENTS OF SHAREHOLDERS[’'’] EQUITY)",
    re.IGNORECASE
)

def clean_section_title(title: str) -> str:
    t = re.sub(r"ITEM\s+\d+\.\s*", "", title, flags=re.IGNORECASE)
    t = re.sub(r"PART\s+[I-V]+\s*", "", t, flags=re.IGNORECASE)
    return t.strip()

def ocr_page_if_needed(page: fitz.Page) -> str:
    """Try OCR if no selectable text and OCR is available."""
    if not OCR_AVAILABLE:
        return ""
    try:
        pix = page.get_pixmap(dpi=300, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

# ------------------------------------------------------------------------------------
# 1) Extract, Clean, Segment
# ------------------------------------------------------------------------------------

def extract_and_clean_text(pdf_path: str, year: int) -> Dict[str, str]:
    """
    Extracts text from a PDF, cleans it, and segments it into logical sections.
    Returns: { "<year> - <section_title>": section_text, ... }
    """
    if not os.path.exists(pdf_path):
        logging.error(f"Missing file: {pdf_path}")
        return {}

    logging.info(f"Extracting: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open {pdf_path}: {e}")
        return {}

    raw_text_parts = []
    for i, page in enumerate(doc):
        try:
            t = page.get_text("text") or ""
            if not t.strip():
                # fallback OCR if available
                t = ocr_page_if_needed(page)
            raw_text_parts.append(t)
        except Exception as e:
            logging.warning(f"Page {i} extract error: {e}")

    full_text = "\n".join(raw_text_parts)
    cleaned_text = simple_clean(full_text)

    # Save cleaned full text per year under data/
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"cleaned_{year}.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # Segment into sections
    sections: Dict[str, str] = {}
    matches = list(SECTION_PATTERN.finditer(cleaned_text))
    if not matches:
        sections[f"{year} - full_text"] = cleaned_text
        return sections

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        section_title = clean_section_title(matches[i].group(0))
        section_text = cleaned_text[start:end].strip()
        sections[f"{year} - {section_title}"] = section_text

    # Save sections per year
    safe_write_json(os.path.join(out_dir, f"sections_{year}.json"), sections)
    return sections

# ------------------------------------------------------------------------------------
# 2) Chunking (two sizes) with IDs & metadata
# ------------------------------------------------------------------------------------

def get_token_length_fn() -> callable:
    """
    Returns a tokenizer-aware length function using the same model family.
    Falls back to len(text) if tokenizer is unavailable.
    """
    try:
        from transformers import AutoTokenizer
        # Use the same model folder you already have under models/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        tok_path = os.path.join(project_root, "models", "all-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        def tok_len(x: str) -> int:
            return len(tokenizer.encode(x, add_special_tokens=False))
        return tok_len
    except Exception:
        logging.warning("Tokenizer not available. Falling back to character length.")
        return len

def split_documents_two_sizes(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    length_fn = get_token_length_fn()

    splitter_small = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20,
        length_function=length_fn, separators=["\n\n", "\n", " ", ""]
    )
    splitter_large = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50,
        length_function=length_fn, separators=["\n\n", "\n", " ", ""]
    )
    return splitter_small.split_documents(documents), splitter_large.split_documents(documents)

def to_chunk_dicts(chunks: List[Document], chunk_size_label: int) -> List[dict]:
    chunk_dicts = []
    for pos, ch in enumerate(chunks):
        cid = str(uuid.uuid4())
        md = dict(ch.metadata or {})
        # Try to derive year and section from "source" if present: "2024 - <section>"
        year = None
        section = None
        src = md.get("source", "")
        m = re.match(r"(\d{4})\s*-\s*(.+)", src)
        if m:
            year = int(m.group(1))
            section = m.group(2).strip()
        chunk_dicts.append({
            "id": cid,
            "page_content": ch.page_content,
            "metadata": {
                "source": src,
                "year": year,
                "section": section,
                "chunk_size": chunk_size_label,
                "position": pos
            }
        })
    return chunk_dicts

# ------------------------------------------------------------------------------------
# 3) Embedding & Indexing (FAISS + BM25)
# ------------------------------------------------------------------------------------

def batch_encode_texts(embedding_model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    embs = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    return embs.cpu().numpy().astype("float32")

def create_faiss_index(chunks: List[dict], embedding_model: SentenceTransformer, faiss_out: str) -> None:
    logging.info("Creating dense vector store (FAISS)...")
    texts = [c["page_content"] for c in chunks]
    embs = batch_encode_texts(embedding_model, texts)

    dim = embs.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    ids = np.arange(len(chunks)).astype("int64")
    index.add_with_ids(embs, ids)
    faiss.write_index(index, faiss_out)
    logging.info(f"FAISS index written to {faiss_out}")

def simple_tokenize_for_bm25(text: str) -> List[str]:
    t = text.lower()
    t = re.sub(r"[^a-z0-9$%.,-]+", " ", t)
    return [w for w in t.split() if w]

def create_bm25_index(chunks: List[dict], bm25_out: str) -> None:
    logging.info("Creating sparse index (BM25)...")
    tokenized_corpus = [simple_tokenize_for_bm25(c["page_content"]) for c in chunks]
    # Keep it simple to persist: we’ll store just the tokenized corpus.
    safe_write_json(bm25_out, tokenized_corpus)
    logging.info(f"BM25 corpus saved to {bm25_out}")

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    pdf_2024_path = os.path.join(project_root, "data", "Apple_2024_10K.pdf")
    pdf_2023_path = os.path.join(project_root, "data", "Apple_2023_10K.pdf")

    # 1) Extract & Clean & Segment (both years)
    sections_2024 = extract_and_clean_text(pdf_2024_path, 2024)
    sections_2023 = extract_and_clean_text(pdf_2023_path, 2023)

    if not sections_2024 or not sections_2023:
        logging.error("Preprocessing failed due to missing/empty sections.")
        raise SystemExit(1)

    # Merge sections into a single dict
    all_sections: Dict[str, str] = {}
    all_sections.update(sections_2024)
    all_sections.update(sections_2023)

    # Save merged sections overview as well
    safe_write_json(os.path.join(data_dir, "sections_all.json"), all_sections)

    # Build LangChain Documents with source metadata intact
    documents: List[Document] = []
    for title, content in all_sections.items():
        if content and content.strip():
            documents.append(Document(page_content=content, metadata={"source": title}))
        else:
            logging.warning(f"Skipped empty section: {title}")

    # 2) Chunk into two sizes
    chunks_small_docs, chunks_large_docs = split_documents_two_sizes(documents)
    logging.info(f"Number of small chunks: {len(chunks_small_docs)}")
    logging.info(f"Number of large chunks: {len(chunks_large_docs)}")

    # Convert to dicts with IDs + metadata
    chunks_small = to_chunk_dicts(chunks_small_docs, 100)
    chunks_large = to_chunk_dicts(chunks_large_docs, 400)

    # Combine both sizes (keeps requirement: at least two chunk sizes)
    combined_chunks = chunks_small + chunks_large

    # Persist combined chunks to data/rag_chunks.json 
    rag_chunks_path = os.path.join(data_dir, "rag_chunks.json")
    safe_write_json(rag_chunks_path, combined_chunks)
    logging.info(f"Saved chunks with metadata to {rag_chunks_path}")

    # Also write a flat id->text map to help debugging
    id_map = {i: {"id": c["id"], "chunk_size": c["metadata"]["chunk_size"]} for i, c in enumerate(combined_chunks)}
    safe_write_json(os.path.join(data_dir, "chunk_id_map.json"), id_map)

    # 3) Embeddings & Indexes
    # Load embedding model from models path
    embedding_model_path = os.path.join(project_root, "models", "all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(embedding_model_path, device=device)

    # FAISS
    faiss_out = os.path.join(data_dir, "faiss_index.bin")               
    create_faiss_index(combined_chunks, embedding_model, faiss_out)

    # BM25
    bm25_out = os.path.join(data_dir, "bm25_corpus.json")               
    create_bm25_index(combined_chunks, bm25_out)

    logging.info("✅ Preprocessing, embedding, and indexing complete. All artifacts are in data/")

