"""
ingest.py — Phase 2: Chunking & Vector Database Indexing Engine
===============================================================
Loads parsed Markdown files, chunks them with an academic-aware strategy,
generates embeddings via a local HuggingFace model, and persists the index
into a local ChromaDB vector store.

Usage:
    python ingest.py
    python ingest.py --reset        # wipe the collection and re-index from scratch
    python ingest.py --model nomic  # use nomic-embed-text via Ollama instead
"""

import argparse
import sys
import time
from pathlib import Path

# ── LlamaIndex core ──────────────────────────────────────────────────────────
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode

# ── Embedding models ─────────────────────────────────────────────────────────
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Vector store ─────────────────────────────────────────────────────────────
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PARSED_MD_DIR   = Path("./data/parsed_markdown")
CHROMA_DB_DIR   = Path("./data/chroma_db")
COLLECTION_NAME = "academic_rag"

# Primary embedding model — runs fully locally, no API key required.
# Switch to "nomic-embed-text" (Ollama) via --model nomic flag.
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Fallback chunking parameters (used after MarkdownNodeParser as a safety net).
CHUNK_SIZE    = 512   # tokens
CHUNK_OVERLAP = 64    # tokens


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    """Print a clearly visible section separator to the terminal."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def _progress(current: int, total: int, label: str = "") -> None:
    """Print an inline progress indicator."""
    pct = int(current / total * 100) if total else 0
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  {label:<50}", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load Parsed Documents
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(md_dir: Path) -> list:
    """
    Load every .md file from *md_dir* using LlamaIndex's SimpleDirectoryReader.
    Each file becomes a LlamaIndex Document object; raw Markdown is preserved
    so the downstream node parsers can exploit structural markers (##, ---, |…|).

    Returns a list of Document objects sorted by filename for deterministic ordering.
    """
    _banner("Step 1 — Loading Parsed Markdown Documents")

    if not md_dir.exists():
        print(f"  [ERROR] Directory not found: {md_dir.resolve()}")
        print("  Make sure Phase 1 (parsing) has been run before ingestion.")
        sys.exit(1)

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        print(f"  [ERROR] No .md files found in: {md_dir.resolve()}")
        sys.exit(1)

    print(f"  Found {len(md_files)} Markdown file(s) in '{md_dir}'\n")

    reader = SimpleDirectoryReader(
        input_dir=str(md_dir),
        required_exts=[".md"],
        recursive=False,            # flat directory — Phase 1 flattens output
        filename_as_id=True,        # doc ID == file path for traceability
    )

    documents = reader.load_data()

    # Log each loaded file
    for i, doc in enumerate(documents, 1):
        file_name = Path(doc.metadata.get("file_name", "unknown")).name
        char_count = len(doc.text)
        print(f"  [{i:>3}] {file_name:<55} ({char_count:,} chars)")

    print(f"\n  ✓ Loaded {len(documents)} document(s) total.")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Academic Chunking Strategy
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list[TextNode]:
    """
    Two-pass academic chunking strategy:

    Pass 1 — MarkdownNodeParser
        Splits on Markdown structural boundaries: section headers (# / ## / ###),
        horizontal rules, and fence blocks.  This keeps each conceptual section
        (Abstract, Methods, Results …) as its own node rather than cutting across
        them arbitrarily.

    Pass 2 — SentenceSplitter fallback
        Any node that survived Pass 1 and is still larger than CHUNK_SIZE tokens
        is further split by the SentenceSplitter, which respects sentence boundaries
        and adds CHUNK_OVERLAP to maintain cross-chunk context.

    Both passes propagate source metadata (file_name, page_label, section header)
    so every node is traceable back to its origin.
    """
    _banner("Step 2 — Academic Chunking Strategy")

    # Pass 1: Markdown-structure-aware splitting
    print("  Pass 1: MarkdownNodeParser (structure-aware split) …")
    md_parser = MarkdownNodeParser()
    md_nodes  = md_parser.get_nodes_from_documents(documents, show_progress=False)
    print(f"          → {len(md_nodes)} nodes after Markdown structural split")

    # Pass 2: Safety-net sentence-level split for oversized nodes
    print(f"  Pass 2: SentenceSplitter fallback (chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) …")
    sentence_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
    )

    final_nodes: list[TextNode] = []
    oversized   = 0

    for node in md_nodes:
        # A rough token estimate: 1 token ≈ 4 chars for English academic text
        estimated_tokens = len(node.text) // 4
        if estimated_tokens > CHUNK_SIZE:
            sub_nodes = sentence_splitter.get_nodes_from_documents(
                [node],  # treat this node as a mini-document
                show_progress=False,
            )
            final_nodes.extend(sub_nodes)
            oversized += 1
        else:
            final_nodes.append(node)

    print(f"          → {oversized} oversized node(s) further split")
    print(f"\n  ✓ Final chunk count: {len(final_nodes)} nodes")

    # Summary statistics
    lengths = [len(n.text) for n in final_nodes]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print(f"     Avg node size : {avg_len:,.0f} chars")
        print(f"     Min / Max     : {min(lengths):,} / {max(lengths):,} chars")

    return final_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Vector Database Connection
# ─────────────────────────────────────────────────────────────────────────────

def init_chroma(chroma_dir: Path, collection_name: str, reset: bool = False):
    """
    Initialise a *persistent* ChromaDB client stored at *chroma_dir*.

    If *reset* is True the existing collection (if any) is deleted so that a
    clean re-index can be performed.  Otherwise the existing collection is loaded
    and new nodes are added incrementally (deduplication is handled by ChromaDB
    via node IDs).

    Returns (chroma_collection, ChromaVectorStore).
    """
    _banner("Step 3 — ChromaDB Vector Store Initialisation")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Persistent path : {chroma_dir.resolve()}")
    print(f"  Collection name : {collection_name}")

    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"  ↻  Existing collection deleted (--reset flag).")
        except Exception:
            pass  # collection did not exist — that's fine

    collection = client.get_or_create_collection(
        name=collection_name,
        # ChromaDB cosine distance matches BGE / nomic embedding conventions
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    print(f"  Existing vectors in store : {existing_count:,}")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    print("  ✓ ChromaDB ready.")
    return collection, vector_store


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Embedding Model Setup
# ─────────────────────────────────────────────────────────────────────────────

def init_embedding_model(model_key: str = "bge") -> HuggingFaceEmbedding:
    """
    Load a local embedding model.  Two presets are supported:

    "bge"   — BAAI/bge-small-en-v1.5   (default)
               ~33 MB, fast CPU inference, strong on academic English.

    "nomic" — nomic-ai/nomic-embed-text-v1
               ~274 MB, higher quality, requires more RAM.

    The model is cached in the default HuggingFace hub directory
    (~/.cache/huggingface/hub) after the first download.
    No API keys or external services are needed at runtime.
    """
    _banner("Step 4 — Embedding Model Initialisation")

    presets = {
        "bge":   {
            "model_name": "BAAI/bge-small-en-v1.5",
            "embed_batch_size": 32,
            "query_instruction": "Represent this sentence for searching relevant passages: ",
        },
        "nomic": {
            "model_name": "nomic-ai/nomic-embed-text-v1",
            "embed_batch_size": 16,
            "query_instruction": "search_query: ",
            "trust_remote_code": True,
        },
    }

    if model_key not in presets:
        print(f"  [WARN] Unknown model key '{model_key}'. Falling back to 'bge'.")
        model_key = "bge"

    cfg = presets[model_key]
    model_name = cfg["model_name"]

    print(f"  Model : {model_name}")
    print(f"  Batch : {cfg['embed_batch_size']} nodes per embedding call")
    print("  Downloading / loading from local cache …")

    start = time.perf_counter()

    # Build kwargs dynamically so we only pass trust_remote_code when needed
    embed_kwargs = {
        "model_name":       model_name,
        "embed_batch_size": cfg["embed_batch_size"],
        "query_instruction": cfg.get("query_instruction", ""),
    }
    if cfg.get("trust_remote_code"):
        embed_kwargs["trust_remote_code"] = True

    embed_model = HuggingFaceEmbedding(**embed_kwargs)

    elapsed = time.perf_counter() - start
    print(f"  ✓ Model loaded in {elapsed:.1f}s")

    # Inject into LlamaIndex global settings so every downstream component uses it
    Settings.embed_model = embed_model
    # Disable the LLM globally — ingestion never needs a generative model
    Settings.llm = None

    return embed_model


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build Index & Persist to Chroma
# ─────────────────────────────────────────────────────────────────────────────

def build_and_persist_index(
    nodes:        list[TextNode],
    vector_store: ChromaVectorStore,
) -> VectorStoreIndex:
    """
    Generate embeddings for every node and persist them to ChromaDB.

    LlamaIndex's VectorStoreIndex handles batching, embedding calls, and writing
    to the vector store automatically.  The StorageContext wires the index to our
    persistent Chroma collection, so there is no separate "save" step — every
    write goes straight to disk.

    Returns the in-memory VectorStoreIndex (useful for an optional smoke-test query).
    """
    _banner("Step 5 — Embedding Generation & Index Persistence")

    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    print(f"  Embedding {len(nodes)} node(s) — this may take a few minutes on CPU …\n")
    t_start = time.perf_counter()

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_ctx,
        show_progress=True,   # LlamaIndex's own tqdm progress bar
    )

    elapsed = time.perf_counter() - t_start
    rate    = len(nodes) / elapsed if elapsed > 0 else 0

    print(f"\n  ✓ Indexing complete in {elapsed:.1f}s  ({rate:.1f} nodes/s)")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Final Report
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(
    collection,
    nodes:        list[TextNode],
    chroma_dir:   Path,
) -> None:
    """Print a summary table so the operator can confirm everything landed correctly."""
    _banner("Ingestion Complete — Summary Report")

    total_in_store = collection.count()

    # Unique source files represented by the new nodes
    source_files = {
        Path(n.metadata.get("file_name", "unknown")).name
        for n in nodes
    }

    print(f"  Source files processed : {len(source_files)}")
    print(f"  Nodes generated        : {len(nodes)}")
    print(f"  Total vectors in store : {total_in_store:,}")
    print(f"  Persisted to           : {chroma_dir.resolve()}")
    print()
    print("  Source files:")
    for fname in sorted(source_files):
        print(f"    • {fname}")

    print()
    print("  The index is ready for Phase 3 (retrieval / querying).")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Chunk & index parsed academic Markdown into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                   # Standard run with BGE embeddings
  python ingest.py --reset           # Wipe existing collection and re-index
  python ingest.py --model nomic     # Use nomic-embed-text instead of BGE
  python ingest.py --md-dir ./custom/path --reset
        """,
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=PARSED_MD_DIR,
        help=f"Directory containing parsed .md files (default: {PARSED_MD_DIR})",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=CHROMA_DB_DIR,
        help=f"ChromaDB persistence directory (default: {CHROMA_DB_DIR})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge",
        choices=["bge", "nomic"],
        help="Embedding model preset: 'bge' (BAAI/bge-small-en-v1.5) or 'nomic' (nomic-embed-text-v1)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete the existing ChromaDB collection before indexing (full re-index)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("  RAG System - Phase 2: Chunking & Indexing Engine")
    print("=" * 70)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    documents = load_documents(args.md_dir)

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    nodes = chunk_documents(documents)

    # ── Step 3: Vector DB ─────────────────────────────────────────────────────
    collection, vector_store = init_chroma(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        reset=args.reset,
    )

    # ── Step 4: Embedding model ───────────────────────────────────────────────
    init_embedding_model(model_key=args.model)

    # ── Step 5: Build & persist ───────────────────────────────────────────────
    build_and_persist_index(nodes=nodes, vector_store=vector_store)

    # ── Step 6: Final report ──────────────────────────────────────────────────
    print_final_report(
        collection=collection,
        nodes=nodes,
        chroma_dir=args.chroma_dir,
    )


if __name__ == "__main__":
    main()
