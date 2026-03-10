# Multimodal RAG (MRAG) Blueprint with Free Models

This document is a practical implementation plan to build a **Multimodal Retrieval-Augmented Generation (MRAG)** system for:
- Text summarization
- Table summarization
- Image summarization/captioning

using **free/open models** that you can run locally.

---

## 1) High-level architecture

1. **Ingestion**
   - Load PDFs, Office docs, HTML, images.
   - Split each source into modality-specific assets:
     - Text blocks
     - Tables
     - Images/figures

2. **Normalization + Summarization (per modality)**
   - Text chunks → text summaries
   - Tables → structured-to-text summaries
   - Images → captions/summaries

3. **Embedding + Indexing**
   - Build embeddings for each summarized asset.
   - Store in vector DB with metadata (`modality`, `source`, `page`, `bbox`, etc.).

4. **Retrieval**
   - User query embedding → top-k candidates across all modalities.
   - Optional reranking for relevance.

5. **Answer generation**
   - Build a grounded prompt from retrieved evidence.
   - Use an instruction model to answer and cite evidence.

---

## 2) Free model stack recommendations

### A. Text summarization (free)
- **Qwen2.5-7B-Instruct** (good quality/latency balance)
- **Llama-3.1-8B-Instruct** (strong baseline)
- **Mistral-7B-Instruct-v0.3** (efficient and reliable)

Use 4-bit quantized variants with `llama.cpp` or `vLLM` where possible.

### B. Table summarization (free)
Two effective approaches:
1. **LLM prompt-based table-to-text** using one of the text models above.
2. **Specialized table model (optional)**:
   - TAPAS/TaPas-like encoders for QA over tables

For MRAG, a strong and simple approach is:
- Convert each table to markdown/CSV excerpt
- Ask LLM for:
  - key columns
  - trends/extremes
  - anomalies
  - compact factual summary

### C. Image summarization/captioning (free)
- **BLIP / BLIP-2** (captioning)
- **LLaVA-1.6 (7B)** for richer image understanding
- **MiniCPM-V** (lightweight VLM options)

Recommended default: LLaVA-1.6-7B (if GPU available), else BLIP-2 for lightweight caption generation.

### D. Embeddings (free)
- **bge-m3** (multilingual, strong retrieval)
- **bge-large-en-v1.5** (English-focused)
- **nomic-embed-text-v1.5** (good open baseline)

### E. Reranker (free, optional but recommended)
- **bge-reranker-large** or **bge-reranker-base**

### F. Vector database (free)
- **Qdrant** (self-hosted)
- **Milvus**
- **Chroma** (fast local prototyping)

---

## 3) End-to-end implementation workflow

## Step 1: Parse documents into modalities
- Text extraction:
  - `pymupdf` / `unstructured` / `docling`
- Table extraction:
  - `camelot` / `tabula-py` / `pdfplumber`
- Image extraction:
  - extract page images or figure crops from PDFs

Persist each unit as:
```json
{
  "id": "doc1_p12_tbl2",
  "modality": "table",
  "raw_content": "...",
  "source": "doc1.pdf",
  "page": 12,
  "metadata": {...}
}
```

## Step 2: Summarize by modality
- **Text**: chunk (400–1000 tokens), summarize each chunk, optionally produce hierarchical summaries.
- **Tables**: serialize table + context and generate factual summary with numeric highlights.
- **Images**: generate caption + dense description + detected entities (if available).

Store both raw + summary:
```json
{
  "id": "doc1_p12_tbl2",
  "summary": "Revenue rises from Q1 to Q4, peaking at ...",
  "raw_content": "<table markdown>",
  "modality": "table"
}
```

## Step 3: Embed and index
Embed **summary text** (and optionally raw text) then upsert into vector DB with metadata filters.

Suggested payload fields:
- `modality`
- `source`
- `page`
- `summary`
- `raw_content_ref`
- `confidence`

## Step 4: Query pipeline
1. User query rewrite (optional)
2. Retrieve top-k across modalities
3. Rerank merged set
4. Construct final grounded context
5. Generate answer with citations and modality tags

---

## 4) Practical prompt templates

## A. Table summarization prompt
```text
You are a data analyst.
Summarize this table factually in 4-6 bullets.
Include: key dimensions, top/bottom values, trends, and anomalies.
Do not invent values.

Table:
{table_markdown}

Context:
{nearby_text}
```

## B. Image summarization prompt
```text
Describe this image for retrieval.
Return:
1) one-sentence caption,
2) detailed description (3-5 lines),
3) key entities/objects,
4) any visible text.
Do not guess details not visible.
```

## C. Final answer prompt (RAG)
```text
Answer the user using only the provided evidence.
If evidence is insufficient, say so.
Cite source IDs in square brackets.

Question: {query}
Evidence:
{retrieved_context}
```

---

## 5) Data schema for multimodal chunks

Use a unified schema so retrieval and filtering stay simple:

```json
{
  "id": "unique_chunk_id",
  "doc_id": "doc_001",
  "modality": "text|table|image",
  "summary": "retrieval text",
  "raw_content": "optional raw excerpt",
  "embedding_model": "bge-m3",
  "source": {
    "file": "report.pdf",
    "page": 8,
    "section": "Results"
  },
  "quality": {
    "summary_confidence": 0.88,
    "ocr_used": true
  },
  "timestamps": {
    "ingested_at": "ISO-8601"
  }
}
```

---

## 6) Recommended minimum viable stack (easy + free)

- Parsing: `unstructured` + `pdfplumber`
- Image captioning: `Salesforce/blip2-opt-2.7b` (or BLIP base)
- Text/table summarization LLM: `Qwen2.5-7B-Instruct`
- Embeddings: `BAAI/bge-m3`
- Vector DB: `Qdrant`
- API layer: `FastAPI`
- Orchestration: simple Python workers + queue (Celery/RQ optional)

---

## 7) Evaluation checklist

Track these before production:

1. **Retrieval quality**
   - Recall@k per modality
   - MRR / nDCG
2. **Summary quality**
   - Factual consistency checks (especially numeric tables)
   - Human spot audits
3. **End-answer quality**
   - Groundedness score
   - Citation accuracy
   - Hallucination rate
4. **Performance**
   - Ingestion time per document
   - Query latency (P50/P95)
   - GPU/CPU memory footprint

---

## 8) 2-week build plan

- **Day 1–2**: ingestion + modality extraction pipeline
- **Day 3–4**: text/table/image summarizers wired
- **Day 5**: embedding + Qdrant indexing
- **Day 6**: retrieval + reranker
- **Day 7**: answer generation API
- **Day 8–9**: eval set creation + metrics
- **Day 10**: prompt tuning and failure analysis
- **Day 11–12**: optimization (batching/quantization/cache)
- **Day 13–14**: hardening + deployment docs

---

## 9) Common pitfalls and fixes

- **Pitfall:** Table summaries hallucinate numbers.
  - **Fix:** force model to quote explicit cells and run numeric consistency checks.

- **Pitfall:** Image captions too generic.
  - **Fix:** two-pass captioning (short + dense) and include OCR text.

- **Pitfall:** One modality dominates retrieval.
  - **Fix:** modality-aware retrieval quotas or score normalization.

- **Pitfall:** Long context degrades answer quality.
  - **Fix:** rerank + compress evidence before final generation.

---

## 10) Next implementation step

Start by implementing a single `ingest_document()` pipeline that outputs standardized chunks for text/table/image and writes them to JSONL. Once stable, plug summarization + embeddings + vector indexing.
