#!/usr/bin/env python3
"""Domain-aware RAG starter for Indian criminal and family law corpora."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def simple_chunk(text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]

    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out


def build_chunks(records: List[Dict]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        text_chunks = simple_chunk(record.get("text", ""))
        for idx, ch in enumerate(text_chunks):
            chunk_id = f"{record['id']}::chunk_{idx}"
            metadata = {
                "source_id": record.get("id"),
                "domain": record.get("domain"),
                "act": record.get("act"),
                "section": record.get("section"),
                "court": record.get("court"),
                "year": record.get("year"),
                "title": record.get("title"),
            }
            chunks.append(Chunk(chunk_id=chunk_id, text=ch, metadata=metadata))
    return chunks


def save_index(index_dir: str, index: faiss.IndexFlatIP, chunks: List[Chunk]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    index_path = Path(index_dir) / "vectors.faiss"
    meta_path = Path(index_dir) / "chunks.json"

    faiss.write_index(index, str(index_path))
    serializable = [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_index(index_dir: str) -> Tuple[faiss.IndexFlatIP, List[Chunk]]:
    index_path = Path(index_dir) / "vectors.faiss"
    meta_path = Path(index_dir) / "chunks.json"

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = [Chunk(chunk_id=i["chunk_id"], text=i["text"], metadata=i["metadata"]) for i in data]
    return index, chunks


def keyword_score(question: str, text: str) -> float:
    q_terms = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    t_terms = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
    if not q_terms:
        return 0.0
    overlap = q_terms.intersection(t_terms)
    return len(overlap) / len(q_terms)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def cmd_build(args: argparse.Namespace) -> None:
    records = load_jsonl(args.input)
    chunks = build_chunks(records)
    model = SentenceTransformer(args.model)

    embeddings = model.encode([c.text for c in chunks], convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings.astype("float32"))

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    save_index(args.index_dir, index, chunks)
    print(f"Built index with {len(chunks)} chunks at: {args.index_dir}")


def metadata_match(metadata: Dict, domain: Optional[str], act: Optional[str]) -> bool:
    if domain and metadata.get("domain") != domain:
        return False
    if act and act.lower() not in str(metadata.get("act", "")).lower():
        return False
    return True


def cmd_ask(args: argparse.Namespace) -> None:
    index, chunks = load_index(args.index_dir)
    model = SentenceTransformer(args.model)

    q_emb = model.encode([args.question], convert_to_numpy=True)
    q_emb = normalize(q_emb.astype("float32"))

    prefiltered_indices = [
        idx
        for idx, c in enumerate(chunks)
        if metadata_match(c.metadata, args.domain, args.act)
    ]

    if not prefiltered_indices:
        print("No chunks matched metadata filters. Try removing --domain/--act.")
        return

    # Gather dense scores against all, then keep only filtered ids.
    dense_scores, dense_ids = index.search(q_emb, min(args.top_k_dense, len(chunks)))
    candidate_scores: List[Tuple[int, float]] = []
    filtered_set = set(prefiltered_indices)

    for idx, score in zip(dense_ids[0], dense_scores[0]):
        if int(idx) in filtered_set:
            candidate_scores.append((int(idx), float(score)))

    if not candidate_scores:
        # fallback to direct filtered list with keyword score only
        candidate_scores = [(idx, 0.0) for idx in prefiltered_indices[: args.top_k_dense]]

    reranked: List[Tuple[int, float]] = []
    for idx, dense in candidate_scores:
        kw = keyword_score(args.question, chunks[idx].text)
        combined = (args.dense_weight * dense) + ((1 - args.dense_weight) * kw)
        reranked.append((idx, combined))

    reranked.sort(key=lambda x: x[1], reverse=True)
    top = reranked[: args.top_k_final]

    print("\nQuestion:")
    print(args.question)
    print("\nTop Context Chunks:\n")

    for rank, (idx, score) in enumerate(top, start=1):
        c = chunks[idx]
        md = c.metadata
        citation = f"{md.get('act')} s.{md.get('section')} ({md.get('year')})"
        print(f"[{rank}] score={score:.4f} | {citation}")
        print(f"title: {md.get('title')}")
        print(c.text)
        print("-" * 80)

    print("\nSuggested answer prompt input:\n")
    context_block = "\n\n".join(
        [f"[{i+1}] {chunks[idx].metadata.get('act')} s.{chunks[idx].metadata.get('section')}: {chunks[idx].text}" for i, (idx, _) in enumerate(top)]
    )
    print("Use only the context below and cite bracket numbers for each proposition.")
    print(context_block)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Indian law RAG (criminal + family)")
    sp = p.add_subparsers(required=True)

    b = sp.add_parser("build", help="Build vector index from JSONL corpus")
    b.add_argument("--input", required=True, help="Path to JSONL corpus")
    b.add_argument("--index-dir", required=True, help="Directory to save index artifacts")
    b.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    b.set_defaults(func=cmd_build)

    a = sp.add_parser("ask", help="Query vector index")
    a.add_argument("--index-dir", required=True, help="Directory containing index artifacts")
    a.add_argument("--question", required=True, help="User question")
    a.add_argument("--domain", choices=["criminal", "family"], default=None)
    a.add_argument("--act", default=None, help="Substring filter for act name")
    a.add_argument("--top-k-dense", type=int, default=20)
    a.add_argument("--top-k-final", type=int, default=5)
    a.add_argument("--dense-weight", type=float, default=0.8)
    a.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    a.set_defaults(func=cmd_ask)

    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
