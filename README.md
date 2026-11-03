# Local Academic RAG System

A fully local Retrieval-Augmented Generation (RAG) pipeline for academic documents.
No API keys required — everything runs on your machine.

## Project Structure

```
RAG System/
├── ingest.py            # Phase 2: Chunking & vector indexing engine
├── requirements.txt     # Python dependencies
├── data/
│   ├── parsed_markdown/ # Phase 1 output: .md files parsed from PDFs
│   └── chroma_db/       # ChromaDB persistent vector store (auto-created)
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Place your parsed `.md` files in `./data/parsed_markdown/`, then run:

```bash
# Standard run (BAAI/bge-small-en-v1.5 embeddings)
python ingest.py

# Full re-index from scratch
python ingest.py --reset

# Higher-quality nomic embeddings (~274 MB)
python ingest.py --model nomic
```

## Embedding Models

Models are downloaded automatically from HuggingFace Hub on first run
and cached locally at `~/.cache/huggingface/hub/`.

| Flag | Model | Size | Notes |
|------|-------|------|-------|
| `--model bge` (default) | `BAAI/bge-small-en-v1.5` | ~33 MB | Fast CPU, strong academic English |
| `--model nomic` | `nomic-ai/nomic-embed-text-v1` | ~274 MB | Higher quality, more RAM |

## Phase Roadmap

- **Phase 1** — PDF parsing → Markdown
- **Phase 2** — Chunking & vector indexing *(this repo)*
- **Phase 3** — Retrieval & query interface
