"""
ingest.py — Phase 2: Chunking & Vector Database Indexing Engine
===============================================================
Loads parsed Markdown files, chunks them with an academic-aware two-pass
strategy, generates embeddings via a local HuggingFace model, and persists
the resulting index into a local ChromaDB vector store.

Workflow:
  1. Load .md files from ./data/parsed_markdown/
  2. Chunk with MarkdownNodeParser (structure-aware) + SentenceSplitter fallback
  3. Connect to persistent ChromaDB at ./data/chroma_db/
  4. Load local HuggingFace embedding model (no API keys needed)
  5. Embed all nodes and persist to Chroma
  6. Print a terminal summary report

Usage:
    python ingest.py                        # Standard run (BGE embeddings)
    python ingest.py --reset                # Wipe collection and re-index
    python ingest.py --model nomic          # Use nomic-embed-text-v1 instead
    python ingest.py --md-dir ./custom/path # Custom markdown directory
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# ── Suppress noisy third-party loggers before any heavy imports ───────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── LlamaIndex core ───────────────────────────────────────────────────────────
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, TextNode

# ── Embedding model ───────────────────────────────────────────────────────────
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Vector store ──────────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.vector_stores.chroma import ChromaVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Project-level defaults  (can be overridden via CLI flags)
# ─────────────────────────────────────────────────────────────────────────────

PARSED_MD_DIR:   Path = Path("./data/parsed_markdown")
CHROMA_DB_DIR:   Path = Path("./data/chroma_db")
COLLECTION_NAME: str  = "academic_rag"

# Chunking parameters
CHUNK_SIZE:    int = 512   # max tokens per node (≈ 4 chars / token for English)
CHUNK_OVERLAP: int = 64    # overlap between adjacent nodes to preserve context

# Embedding model presets
EMBED_PRESETS: dict[str, dict] = {
    "bge": {
        "model_name":        "BAAI/bge-small-en-v1.5",
        "embed_batch_size":  32,
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "description":       "~33 MB · Fast CPU · Strong academic English",
    },
    "nomic": {
        "model_name":        "nomic-ai/nomic-embed-text-v1",
        "embed_batch_size":  16,
        "query_instruction": "search_query: ",
        "trust_remote_code": True,
        "description":       "~274 MB · Higher quality · More RAM required",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 70


def _banner(title: str) -> None:
    """Print a clearly visible section header."""
    print(f"\n{'=' * WIDTH}")
    print(f"  {title}")
    print(f"{'=' * WIDTH}")


def _ok(msg: str) -> None:
    print(f"  OK {msg}")


def _info(msg: str) -> None:
    print(f"  - {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN]  {msg}", file=sys.stderr)


def _fatal(msg: str) -> None:
    print(f"\n  [FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load Parsed Documents
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(md_dir: Path) -> list[Document]:
    """
    Load every .md file from *md_dir* using LlamaIndex's SimpleDirectoryReader.

    Each file becomes a LlamaIndex Document; raw Markdown content is preserved
    so that downstream parsers can exploit structural cues (##, ---, |…| tables).
    Documents are sorted by filename for deterministic, reproducible ordering.

    Raises SystemExit if the directory is missing or contains no .md files.
    """
    _banner("Step 1 — Loading Parsed Markdown Documents")

    if not md_dir.exists():
        _fatal(
            f"Directory not found: {md_dir.resolve()}\n"
            "  Run Phase 1 (parsing) first to populate this directory."
        )

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        _fatal(
            f"No .md files found in: {md_dir.resolve()}\n"
            "  Place your parsed Markdown files there and retry."
        )

    _info(f"Found {len(md_files)} Markdown file(s) in '{md_dir}'")
    print()

    reader = SimpleDirectoryReader(
        input_dir=str(md_dir),
        required_exts=[".md"],
        recursive=False,        # Phase 1 produces a flat directory
        filename_as_id=True,    # doc.doc_id == file path for traceability
    )

    documents: list[Document] = reader.load_data()

    # Per-file log
    for idx, doc in enumerate(documents, start=1):
        fname      = Path(doc.metadata.get("file_name", "unknown")).name
        char_count = len(doc.text)
        print(f"  [{idx:>3}] {fname:<52} {char_count:>10,} chars")

    print()
    _ok(f"Loaded {len(documents)} document(s).")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Academic Chunking Strategy
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[BaseNode]:
    """
    Two-pass academic chunking strategy designed to preserve document structure.

    Pass 1 — MarkdownNodeParser
      Splits on Markdown structural boundaries: ATX headers (# / ## / ###),
      horizontal rules (---), and fenced code/table blocks.  Each logical
      section (Abstract, Introduction, Methods, Results, …) becomes its own
      node, ensuring that semantically coherent content is never split in the
      middle of a thought.

    Pass 2 — SentenceSplitter (overflow guard)
      Any node from Pass 1 that still exceeds CHUNK_SIZE tokens (rough
      estimate: 1 token ≈ 4 characters) is further split respecting sentence
      boundaries, with CHUNK_OVERLAP characters of context overlap to prevent
      information loss at chunk boundaries.

    Both passes propagate source metadata (file_name, page_label, header path)
    so every chunk can be traced back to its origin document and section.

    Returns a flat list of TextNode objects ready for embedding.
    """
    _banner("Step 2 — Academic Chunking Strategy")

    # ── Pass 1: Markdown structure-aware split ────────────────────────────────
    _info("Pass 1: MarkdownNodeParser (split on headers / HR / tables) …")
    md_parser  = MarkdownNodeParser()
    md_nodes   = md_parser.get_nodes_from_documents(documents, show_progress=False)
    _info(f"        → {len(md_nodes)} nodes after structural split")

    # ── Pass 2: Sentence-level overflow guard ─────────────────────────────────
    _info(
        f"Pass 2: SentenceSplitter fallback "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) …"
    )
    sentence_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
    )

    final_nodes: list[BaseNode] = []
    overflow_count = 0

    for node in md_nodes:
        # Rough token estimate: 4 chars ≈ 1 token for English academic prose
        approx_tokens = len(node.text) // 4

        if approx_tokens > CHUNK_SIZE:
            # Treat this oversized node as a mini-document for the splitter
            sub_nodes = sentence_splitter.get_nodes_from_documents(
                [node],
                show_progress=False,
            )
            final_nodes.extend(sub_nodes)
            overflow_count += 1
        else:
            final_nodes.append(node)

    _info(f"        → {overflow_count} oversized node(s) split further")

    # ── Statistics ────────────────────────────────────────────────────────────
    print()
    lengths = [len(n.text) for n in final_nodes]
    if lengths:
        avg = sum(lengths) / len(lengths)
        _ok(
            f"Final chunk count: {len(final_nodes)} nodes  "
            f"(avg {avg:,.0f} chars · min {min(lengths):,} · max {max(lengths):,})"
        )
    else:
        _ok("Final chunk count: 0 nodes")

    return final_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Vector Database Connection
# ─────────────────────────────────────────────────────────────────────────────

def init_chroma(
    chroma_dir:      Path,
    collection_name: str,
    reset:           bool = False,
) -> tuple[chromadb.Collection, ChromaVectorStore]:
    """
    Initialise a *persistent* ChromaDB client stored at *chroma_dir*.

    Persistence model
    -----------------
    ChromaDB's PersistentClient writes every mutation to disk immediately via
    SQLite + HNSW on-disk files.  There is no explicit .persist() call needed —
    data is safe as soon as the embedding call returns.

    Reset flag
    ----------
    When *reset=True* the existing collection is deleted before re-creation,
    enabling a clean full re-index without leaving stale vectors behind.
    When *reset=False* the collection is opened in append mode; duplicate node
    IDs are silently ignored by ChromaDB (upsert semantics).

    Returns
    -------
    (chromadb.Collection, ChromaVectorStore)  — both wired to the same store.
    """
    _banner("Step 3 — ChromaDB Vector Store Initialisation")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    _info(f"Persistent path : {chroma_dir.resolve()}")
    _info(f"Collection name : {collection_name}")

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(collection_name)
            _info("Existing collection deleted (--reset flag active).")
        except Exception:
            _info("No existing collection found to delete — starting fresh.")

    collection = client.get_or_create_collection(
        name=collection_name,
        # Cosine distance is standard for normalised BGE / nomic embeddings
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    _info(f"Vectors already in store : {existing:,}")
    _ok("ChromaDB ready.")
    return collection, ChromaVectorStore(chroma_collection=collection)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Embedding Model Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_embedding_model(model_key: str = "bge") -> HuggingFaceEmbedding:
    """
    Load a fully local embedding model from the HuggingFace Hub.

    Models are downloaded once and cached at ~/.cache/huggingface/hub/.
    No API key or internet connection is needed after the initial download.

    Presets
    -------
    "bge"   — BAAI/bge-small-en-v1.5  (~33 MB)
               Fast CPU inference, state-of-the-art on BEIR academic benchmarks.
               Default choice for most workstations.

    "nomic" — nomic-ai/nomic-embed-text-v1  (~274 MB)
               Higher quality at the cost of ~8× more RAM and slower encoding.
               Requires trust_remote_code=True.

    The loaded model is injected into LlamaIndex's global Settings so every
    component in the pipeline automatically uses it without further wiring.
    The LLM is explicitly disabled because ingestion never needs text generation.
    """
    _banner("Step 4 — Embedding Model Initialisation")

    if model_key not in EMBED_PRESETS:
        _warn(f"Unknown model key '{model_key}'. Falling back to 'bge'.")
        model_key = "bge"

    cfg        = EMBED_PRESETS[model_key]
    model_name = cfg["model_name"]

    _info(f"Model    : {model_name}")
    _info(f"Profile  : {cfg['description']}")
    _info(f"Batch    : {cfg['embed_batch_size']} nodes per embedding call")
    _info("Loading model (first run downloads to HuggingFace cache) …")

    t_start = time.perf_counter()

    # Auto-detect CUDA and use GPU if available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        _info(f"Device   : GPU — {gpu_name} ({vram_gb} GB VRAM)")
        # Larger batch on GPU for faster throughput
        cfg["embed_batch_size"] = 128
    else:
        _info("Device   : CPU (no CUDA GPU detected)")

    # Build kwargs; only include trust_remote_code when the preset requires it
    kwargs: dict = {
        "model_name":        model_name,
        "embed_batch_size":  cfg["embed_batch_size"],
        "query_instruction": cfg.get("query_instruction", ""),
        "device":            device,
    }
    if cfg.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    embed_model = HuggingFaceEmbedding(**kwargs)

    elapsed = time.perf_counter() - t_start
    _ok(f"Model loaded in {elapsed:.1f}s")

    # Wire into LlamaIndex globals
    Settings.embed_model = embed_model
    Settings.llm         = None   # no LLM needed during ingestion

    return embed_model


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build Index & Persist to ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def build_and_persist_index(
    nodes:        list[BaseNode],
    vector_store: ChromaVectorStore,
) -> VectorStoreIndex:
    """
    Embed all nodes and persist the index to ChromaDB.

    LlamaIndex's VectorStoreIndex handles:
      - Batching embedding calls according to embed_batch_size
      - Calling the embedding model for each batch
      - Writing (id, embedding, metadata, text) tuples to ChromaDB via upsert

    Because ChromaVectorStore is backed by a PersistentClient, every write is
    flushed to disk immediately — there is no separate "save" or "commit" step.

    The returned VectorStoreIndex object can be used directly for smoke-test
    queries in Phase 3 without reloading from disk.
    """
    _banner("Step 5 — Embedding Generation & Index Persistence")

    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    _info(f"Embedding {len(nodes)} node(s) — may take several minutes on CPU …")
    print()

    t_start = time.perf_counter()

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_ctx,
        show_progress=True,   # renders LlamaIndex's built-in tqdm progress bar
    )

    elapsed = time.perf_counter() - t_start
    rate    = len(nodes) / elapsed if elapsed > 0 else 0.0

    print()  # newline after tqdm bar
    _ok(f"Indexing complete in {elapsed:.1f}s  ({rate:.1f} nodes/s)")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Terminal Summary Report
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(
    collection:  chromadb.Collection,
    nodes:       list[BaseNode],
    chroma_dir:  Path,
    model_key:   str,
) -> None:
    """
    Print a formatted summary confirming what was indexed and where it lives.
    Gives the operator a quick sanity-check before moving to Phase 3.
    """
    _banner("Ingestion Complete — Summary Report")

    total_in_store = collection.count()
    source_files   = {
        Path(n.metadata.get("file_name", "unknown")).name
        for n in nodes
    }

    cfg = EMBED_PRESETS.get(model_key, EMBED_PRESETS["bge"])

    rows = [
        ("Source files processed", str(len(source_files))),
        ("Nodes generated",        str(len(nodes))),
        ("Total vectors in store", f"{total_in_store:,}"),
        ("Embedding model",        cfg["model_name"]),
        ("Persisted to",           str(chroma_dir.resolve())),
    ]

    col_w = max(len(r[0]) for r in rows) + 2
    for label, value in rows:
        print(f"  {label:<{col_w}}: {value}")

    print()
    print("  Source files:")
    for fname in sorted(source_files):
        print(f"    - {fname}")

    print()
    _ok("Index is ready for Phase 3 (retrieval & querying).")
    print(f"{'=' * WIDTH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="Phase 2: Chunk & index parsed academic Markdown into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap_dedent("""
        Examples:
          python ingest.py
          python ingest.py --reset
          python ingest.py --model nomic
          python ingest.py --md-dir ./papers --chroma-dir ./db --reset
        """),
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=PARSED_MD_DIR,
        metavar="PATH",
        help=f"Directory with parsed .md files  (default: {PARSED_MD_DIR})",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=CHROMA_DB_DIR,
        metavar="PATH",
        help=f"ChromaDB persistence directory   (default: {CHROMA_DB_DIR})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        metavar="NAME",
        help=f"ChromaDB collection name         (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge",
        choices=list(EMBED_PRESETS.keys()),
        help="Embedding model preset  (default: bge)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete the existing collection before indexing (full re-index)",
    )
    return parser.parse_args()


# textwrap.dedent alias so the import stays at the top even if we add it later
def textwrap_dedent(s: str) -> str:
    import textwrap
    return textwrap.dedent(s)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"\n{'=' * WIDTH}")
    print("  RAG System — Phase 2: Chunking & Indexing Engine")
    print(f"{'=' * WIDTH}")
    _info(f"Markdown dir  : {args.md_dir.resolve()}")
    _info(f"ChromaDB dir  : {args.chroma_dir.resolve()}")
    _info(f"Collection    : {args.collection}")
    _info(f"Embed model   : {EMBED_PRESETS[args.model]['model_name']}")
    _info(f"Reset mode    : {'YES — existing data will be wiped' if args.reset else 'NO  — incremental append'}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    documents = load_documents(args.md_dir)

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    nodes = chunk_documents(documents)

    # ── 3. Vector DB ──────────────────────────────────────────────────────────
    collection, vector_store = init_chroma(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        reset=args.reset,
    )

    # ── 4. Embedding model ────────────────────────────────────────────────────
    init_embedding_model(model_key=args.model)

    # ── 5. Build & persist ────────────────────────────────────────────────────
    build_and_persist_index(nodes=nodes, vector_store=vector_store)

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print_final_report(
        collection=collection,
        nodes=nodes,
        chroma_dir=args.chroma_dir,
        model_key=args.model,
    )


if __name__ == "__main__":
    main()

