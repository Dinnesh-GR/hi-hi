# Indian Law RAG Seed Data (Criminal + Family Law)

This repository now contains a **starter dataset** for building an Indian-law-focused Retrieval-Augmented Generation (RAG) system.

## What is included

- `data/indian_law_rag_seed.jsonl`: Seed corpus metadata and short snippets for:
  - Criminal law (substantive, procedural, evidence, special criminal statutes)
  - Family law (marriage, divorce, maintenance, adoption, succession, domestic violence)
- `scripts/validate_dataset.py`: Lightweight schema validator for the seed JSONL.

## Suggested next steps for your RAG pipeline

1. Use this seed list to fetch authoritative full texts from official sources.
2. Normalize each document into chunkable plain text with citation metadata.
3. Chunk by logical units (sections/articles/paragraphs).
4. Create embeddings and store with metadata fields such as:
   - `domain` (`criminal_law` / `family_law`)
   - `doc_type` (`statute` / `case_law` / `commentary`)
   - `court`, `year`, `act`, `section`, `language`
5. Add reranking and citation-grounded response generation.

## Validation

Run:

```bash
python3 scripts/validate_dataset.py data/indian_law_rag_seed.jsonl
```

