# Indian Law RAG (Criminal Law + Family Law)

This repository provides a **practical starter implementation** of a Retrieval-Augmented Generation (RAG) pipeline tailored for Indian legal use cases, specifically:

- **Criminal law** (e.g., BNS/Bharatiya Nyaya Sanhita, BNSS, constitutional safeguards, precedent)
- **Family law** (e.g., Hindu Marriage Act, Special Marriage Act, maintenance, divorce, custody, domestic violence)

> ⚠️ This project is a technical template and **not legal advice**.

## 1) What this includes

- A domain-aware ingest + chunk + embed + retrieve pipeline in `rag_indian_law.py`
- Metadata-first retrieval for legal filters (`domain`, `act`, `section`, `court`, `year`)
- Hybrid reranking approach (vector similarity + keyword overlap)
- Citation-friendly answer assembly
- Sample corpus records in `data/legal_corpus.jsonl`

## 2) Suggested corpus design

Use JSONL with one legal unit per line. Typical units:

- Bare act section text
- Headnotes / ratio excerpts from judgments
- Official FAQ or rules

Each record should contain at least:

```json
{
  "id": "hma_13_001",
  "domain": "family",
  "act": "Hindu Marriage Act, 1955",
  "section": "13",
  "court": "-",
  "year": 1955,
  "title": "Divorce",
  "text": "Any marriage solemnized ... may, on a petition ... be dissolved ..."
}
```

### Domain tags

- `criminal`
- `family`

## 3) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Build the vector index

```bash
python rag_indian_law.py build \
  --input data/legal_corpus.jsonl \
  --index-dir ./artifacts/index
```

## 5) Query examples

Criminal-law query:

```bash
python rag_indian_law.py ask \
  --index-dir ./artifacts/index \
  --question "What are key ingredients of murder and culpable homicide distinctions in Indian criminal law?" \
  --domain criminal
```

Family-law query:

```bash
python rag_indian_law.py ask \
  --index-dir ./artifacts/index \
  --question "What grounds are available for divorce under Hindu Marriage Act and what evidence is usually relevant?" \
  --domain family
```

## 6) Production hardening checklist

1. **Authoritative sources only**: ingest from official gazettes, eCourts, SCC-compatible licensed data.
2. **Versioning**: keep amendment date and effective dates for each section.
3. **Jurisdiction controls**: Supreme Court vs High Court precedential handling.
4. **Prompt policy**: require citation of act/section/case for every legal proposition.
5. **Human-in-the-loop**: legal expert review before any user-facing deployment.
6. **Audit logs**: store retrieval IDs and generated outputs for compliance.

## 7) Typical architecture

1. Load JSONL records
2. Chunk each record with metadata inheritance
3. Embed chunks (`sentence-transformers`)
4. Persist vectors + metadata via FAISS + local JSON
5. At query time:
   - optional metadata prefilter (`domain`, `act`, etc.)
   - dense retrieval
   - lightweight rerank
   - construct citation-grounded context
6. Send context to your LLM of choice for final answer generation

## 8) Legal and ethical note

This pipeline can improve research workflows, but final legal conclusions should be validated by qualified advocates and current statutory text.
