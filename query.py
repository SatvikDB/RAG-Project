"""
query.py - Phase 3: Retrieval & Synthesis Engine
=================================================
Reconnects to the ChromaDB index built in Phase 2, runs a hybrid
BM25 + dense-vector search over it, and synthesises a cited answer
using a local Ollama LLM.

Public API
----------
    from query import run_academic_query
    result = run_academic_query("What is the role of attention in transformers?")
    print(result.response)

CLI
---
    python query.py "What is the role of attention in transformers?"
    python query.py --model mistral --top-k 8 "Explain gradient descent"
"""

import argparse
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

# ── LlamaIndex core ──────────────────────────────────────────────────────────
from llama_index.core import QueryBundle, Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.prompts import PromptTemplate

# ── BM25 retriever ───────────────────────────────────────────────────────────
from llama_index.retrievers.bm25 import BM25Retriever

# ── Embedding model (must match Phase 2) ─────────────────────────────────────
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Ollama LLM ───────────────────────────────────────────────────────────────
from llama_index.llms.ollama import Ollama

# ── Vector store ─────────────────────────────────────────────────────────────
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (mirrors Phase 2 constants — keep in sync)
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_DB_DIR   = Path("./data/chroma_db")
COLLECTION_NAME = "academic_rag"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
EMBED_BATCH     = 32

DEFAULT_LLM_MODEL  = "llama3.1:8b"   # swap to "mistral", "gemma2", etc.
OLLAMA_BASE_URL    = "http://localhost:11434"
OLLAMA_TIMEOUT_SEC = 120.0

DEFAULT_TOP_K          = 6    # nodes retrieved per retriever
SIMILARITY_CUTOFF      = 0.0   # RRF scores are small by nature; no cutoff needed
HYBRID_VECTOR_WEIGHT   = 0.6  # weight for dense retriever in score fusion
HYBRID_BM25_WEIGHT     = 0.4  # weight for BM25 retriever in score fusion


# ─────────────────────────────────────────────────────────────────────────────
# Citation-aware system prompt
# ─────────────────────────────────────────────────────────────────────────────

CITATION_PROMPT_TMPL = PromptTemplate(
    textwrap.dedent("""\
    You are an expert academic research assistant. Your task is to answer the
    user's question using ONLY the information found in the provided source
    passages below. Do NOT use any external knowledge or assumptions.

    CITATION RULES (mandatory):
    - For every distinct factual claim or summary statement you make, append a
      citation immediately after it in this exact format:
          [<source_filename>, Sec. <section_or_header>]
      Example: "The attention mechanism scales as O(n²) in memory.
               [attention_is_all_you_need.pdf, Sec. 3.2]"
    - If a passage has no identifiable section header, cite it as:
          [<source_filename>, para. <approximate_paragraph_number>]
    - If a claim is supported by multiple passages, list all citations:
          [paper_a.pdf, Sec. 2] [paper_b.pdf, Sec. 4.1]
    - If the provided passages do NOT contain enough information to answer the
      question, you MUST respond with exactly:
          "The provided documents do not contain sufficient information to
           answer this question."
      Do not speculate or fill gaps with outside knowledge.

    SOURCE PASSAGES:
    ----------------
    {context_str}
    ----------------

    USER QUESTION: {query_str}

    ANSWER (with mandatory inline citations):
    """)
)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever  (BM25 + dense vector, reciprocal-rank fusion)
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Combines a dense VectorIndexRetriever with a sparse BM25Retriever using
    Reciprocal Rank Fusion (RRF).

    RRF score for node i:  sum_r( weight_r / (rank_i_in_r + k) )
    where k=60 is the standard RRF smoothing constant.

    This prevents any single very-high-scoring result from dominating and
    ensures rare academic terms found only by BM25 still surface when they
    appear low in the vector ranking.
    """

    RRF_K = 60  # standard RRF smoothing constant

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever:   BM25Retriever,
        vector_weight:    float = HYBRID_VECTOR_WEIGHT,
        bm25_weight:      float = HYBRID_BM25_WEIGHT,
        top_k:            int   = DEFAULT_TOP_K,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever   = bm25_retriever
        self._vector_weight    = vector_weight
        self._bm25_weight      = bm25_weight
        self._top_k            = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # ── Run both retrievers in parallel (within single thread) ────────────
        vector_nodes: list[NodeWithScore] = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes:   list[NodeWithScore] = self._bm25_retriever.retrieve(query_bundle)

        # ── Build node_id → NodeWithScore maps ───────────────────────────────
        node_map: dict[str, NodeWithScore] = {}
        scores:   dict[str, float]         = {}

        def _apply_rrf(ranked_nodes: list[NodeWithScore], weight: float) -> None:
            for rank, nws in enumerate(ranked_nodes, start=1):
                nid = nws.node.node_id
                rrf_score = weight / (rank + self.RRF_K)
                if nid not in node_map:
                    node_map[nid] = nws
                    scores[nid]   = 0.0
                scores[nid] += rrf_score

        _apply_rrf(vector_nodes, self._vector_weight)
        _apply_rrf(bm25_nodes,   self._bm25_weight)

        # ── Sort by fused score, return top-k ────────────────────────────────
        sorted_ids = sorted(scores, key=lambda nid: scores[nid], reverse=True)
        fused: list[NodeWithScore] = []
        for nid in sorted_ids[: self._top_k]:
            nws = node_map[nid]
            fused.append(NodeWithScore(node=nws.node, score=scores[nid]))

        return fused


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Reconnect to ChromaDB index
# ─────────────────────────────────────────────────────────────────────────────

def load_index(
    chroma_dir:      Path = CHROMA_DB_DIR,
    collection_name: str  = COLLECTION_NAME,
) -> VectorStoreIndex:
    """
    Reconnect to the persistent ChromaDB collection created by ingest.py
    and wrap it in a LlamaIndex VectorStoreIndex (read-only).

    Raises
    ------
    FileNotFoundError   if the chroma_dir does not exist.
    RuntimeError        if the collection is empty (ingestion never ran).
    """
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {chroma_dir.resolve()}\n"
            "Run ingest.py first to build the index."
        )

    client     = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    count = collection.count()
    if count == 0:
        raise RuntimeError(
            f"Collection '{collection_name}' exists but contains 0 vectors.\n"
            "Run ingest.py to populate the index before querying."
        )

    print(f"  [DB]  Connected to '{collection_name}'  ({count:,} vectors)")

    vector_store  = ChromaVectorStore(chroma_collection=collection)
    storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)
    index         = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_ctx,
    )
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Embedding model (must match Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def init_embed_model(model_name: str = EMBED_MODEL) -> HuggingFaceEmbedding:
    """
    Load the same embedding model used during ingestion so that query vectors
    live in the same space as the stored document vectors.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        embed_batch_size=EMBED_BATCH,
        query_instruction=(
            "Represent this sentence for searching relevant passages: "
        ),
    )
    Settings.embed_model = embed_model
    Settings.llm         = None   # overridden below once Ollama is confirmed
    print(f"  [EMB] Embedding model: {model_name}")
    return embed_model


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Ollama LLM with connectivity check
# ─────────────────────────────────────────────────────────────────────────────

def init_llm(
    model_name:  str   = DEFAULT_LLM_MODEL,
    base_url:    str   = OLLAMA_BASE_URL,
    timeout:     float = OLLAMA_TIMEOUT_SEC,
) -> Ollama:
    """
    Initialise the Ollama LLM and verify the server is reachable before
    handing it back to the caller.

    Raises
    ------
    ConnectionError   if the Ollama server is not running or unreachable.
    RuntimeError      if the requested model is not available locally.
    """
    import urllib.request
    import urllib.error
    import json

    # ── Liveness check ────────────────────────────────────────────────────────
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}.\n"
            "Make sure Ollama is running:  ollama serve\n"
            f"Original error: {exc}"
        ) from exc

    # ── Model availability check ──────────────────────────────────────────────
    available_models = [m["name"] for m in payload.get("models", [])]
    # Normalise: "llama3.1:8b" and "llama3.1" should both match
    model_base = model_name.split(":")[0]
    matched = any(
        m == model_name or m.startswith(model_base)
        for m in available_models
    )
    if not matched:
        avail_str = ", ".join(available_models) or "(none)"
        raise RuntimeError(
            f"Model '{model_name}' is not available in your local Ollama.\n"
            f"Available models: {avail_str}\n"
            f"Pull it with:  ollama pull {model_name}"
        )

    llm = Ollama(model=model_name, base_url=base_url, request_timeout=timeout)
    Settings.llm = llm
    print(f"  [LLM] Ollama model  : {model_name}  ({base_url})")
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build hybrid retriever
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_retriever(
    index:  VectorStoreIndex,
    top_k:  int = DEFAULT_TOP_K,
) -> HybridRetriever:
    """
    Construct the two-arm hybrid retriever:

    Dense arm  — VectorIndexRetriever queries Chroma with the embedded query
                 vector; captures semantic similarity even when exact terms
                 differ between query and document.

    Sparse arm — BM25Retriever runs over all nodes stored in the index;
                 captures exact-match term frequency, which is critical for
                 academic acronyms (e.g. "BLEU", "RLHF", "LoRA") and
                 domain-specific jargon that may appear infrequently.

    Both arms retrieve `top_k` candidates independently; the HybridRetriever
    fuses them via Reciprocal Rank Fusion and returns the best `top_k` overall.
    """
    # Dense retriever — hits the Chroma HNSW index
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # Sparse retriever — needs the raw nodes (fetched from Chroma store)
    # We pull all nodes from the docstore; BM25Retriever builds its index
    # in-memory at query time (fast for typical academic corpora < 50k nodes).
    all_nodes: list[TextNode] = _fetch_all_nodes(index)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes,
        similarity_top_k=top_k,
    )

    print(
        f"  [RET] Hybrid retriever ready  "
        f"(vector={HYBRID_VECTOR_WEIGHT}, BM25={HYBRID_BM25_WEIGHT}, "
        f"top_k={top_k}, corpus={len(all_nodes):,} nodes)"
    )

    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        top_k=top_k,
    )


def _fetch_all_nodes(index: VectorStoreIndex) -> list[TextNode]:
    """
    Pull every stored node from the underlying ChromaDB collection so the
    BM25 retriever can build its in-memory inverted index over the full corpus.
    """
    chroma_vs: ChromaVectorStore = index.storage_context.vector_store  # type: ignore[attr-defined]
    collection = chroma_vs._collection  # ChromaVectorStore internal attribute

    # Retrieve in batches to avoid Chroma's default 41,666-item page limit
    BATCH = 5_000
    offset = 0
    raw_nodes: list[TextNode] = []

    while True:
        batch = collection.get(
            limit=BATCH,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids       = batch.get("ids", [])
        docs      = batch.get("documents", [])
        metas     = batch.get("metadatas", [])

        if not ids:
            break

        for nid, text, meta in zip(ids, docs, metas):
            raw_nodes.append(
                TextNode(
                    id_=nid,
                    text=text or "",
                    metadata=meta or {},
                )
            )
        offset += len(ids)
        if len(ids) < BATCH:
            break

    return raw_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Assemble the query engine
# ─────────────────────────────────────────────────────────────────────────────

def build_query_engine(
    index:  VectorStoreIndex,
    llm:    Ollama,
    top_k:  int = DEFAULT_TOP_K,
) -> RetrieverQueryEngine:
    """
    Wire hybrid retriever + similarity filter + citation prompt + Ollama LLM
    into a single RetrieverQueryEngine.
    """
    hybrid_retriever = build_hybrid_retriever(index=index, top_k=top_k)

    # Post-retrieval filter: discard any node whose fused RRF score falls below
    # the cutoff — prevents low-quality context from corrupting the answer.
    similarity_filter = SimilarityPostprocessor(
        similarity_cutoff=SIMILARITY_CUTOFF
    )

    engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[similarity_filter],
        text_qa_template=CITATION_PROMPT_TMPL,
        llm=llm,
        streaming=False,
    )
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class QueryResult:
    """Structured return value from run_academic_query()."""

    def __init__(
        self,
        response:       str,
        source_nodes:   list[NodeWithScore],
        elapsed_sec:    float,
    ) -> None:
        self.response     = response
        self.source_nodes = source_nodes
        self.elapsed_sec  = elapsed_sec

    # ── Convenience display ───────────────────────────────────────────────────
    def print_answer(self) -> None:
        _section("Answer")
        print(textwrap.fill(self.response, width=88))

    def print_sources(self) -> None:
        _section(f"Retrieved Source Nodes  ({len(self.source_nodes)} total)")
        for i, nws in enumerate(self.source_nodes, 1):
            meta      = nws.node.metadata
            fname     = meta.get("file_name", "unknown")
            header    = meta.get("header_path", meta.get("section", "—"))
            score     = nws.score or 0.0
            snippet   = nws.node.text[:160].replace("\n", " ")
            print(
                f"  [{i}] {Path(fname).name}  |  {header}  |  score={score:.4f}\n"
                f"      \"{snippet}…\"\n"
            )

    def __repr__(self) -> str:
        return (
            f"QueryResult(elapsed={self.elapsed_sec:.1f}s, "
            f"nodes={len(self.source_nodes)}, "
            f"response_len={len(self.response)})"
        )


def run_academic_query(
    user_query:      str,
    llm_model:       str   = DEFAULT_LLM_MODEL,
    top_k:           int   = DEFAULT_TOP_K,
    chroma_dir:      Path  = CHROMA_DB_DIR,
    collection_name: str   = COLLECTION_NAME,
) -> QueryResult:
    """
    End-to-end academic RAG query.

    Parameters
    ----------
    user_query      : Natural-language question to answer from the corpus.
    llm_model       : Ollama model tag, e.g. "llama3.1:8b" or "mistral".
    top_k           : Number of candidate nodes retrieved per retriever arm.
    chroma_dir      : Path to the ChromaDB persistence directory.
    collection_name : Name of the ChromaDB collection to query.

    Returns
    -------
    QueryResult with .response (str), .source_nodes (list), .elapsed_sec (float).

    Raises
    ------
    FileNotFoundError   — ChromaDB directory not found (ingest.py not run).
    RuntimeError        — Collection is empty.
    ConnectionError     — Ollama server is not reachable.
    RuntimeError        — Requested LLM model not pulled in Ollama.
    """
    print(f"\n  [QRY] Query: \"{user_query}\"")

    # Step 2: Embed model FIRST — must be set before index loads
    init_embed_model()

    # Step 3: LLM
    llm = init_llm(model_name=llm_model)

    # Step 1: DB (after embed model is wired into Settings)
    index = load_index(chroma_dir=chroma_dir, collection_name=collection_name)

    # Step 4+5: Engine
    engine = build_query_engine(index=index, llm=llm, top_k=top_k)

    # Execute
    print("  [QRY] Running hybrid search + synthesis …")
    t0 = time.perf_counter()
    response_obj = engine.query(user_query)
    elapsed = time.perf_counter() - t0
    print(f"  [QRY] Done in {elapsed:.1f}s")

    return QueryResult(
        response=str(response_obj),
        source_nodes=response_obj.source_nodes,
        elapsed_sec=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Hybrid academic RAG query engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python query.py "What is the role of attention in transformers?"
          python query.py --model mistral "Explain the BLEU metric"
          python query.py --top-k 10 "Compare LoRA and full fine-tuning"
        """),
    )
    parser.add_argument("query", help="Natural-language question to answer.")
    parser.add_argument(
        "--model", default=DEFAULT_LLM_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_LLM_MODEL})"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Nodes retrieved per retriever arm (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--chroma-dir", type=Path, default=CHROMA_DB_DIR,
        help=f"ChromaDB directory (default: {CHROMA_DB_DIR})"
    )
    parser.add_argument(
        "--collection", default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME})"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    _section("RAG System - Phase 3: Academic Query Engine")

    try:
        result = run_academic_query(
            user_query=args.query,
            llm_model=args.model,
            top_k=args.top_k,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
        )
    except FileNotFoundError as exc:
        print(f"\n  [ERROR] Index not found:\n  {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n  [ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        print(f"\n  [ERROR] Ollama unreachable:\n  {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.", file=sys.stderr)
        sys.exit(0)

    result.print_answer()
    result.print_sources()

    _section("Summary")
    print(f"  Elapsed    : {result.elapsed_sec:.1f}s")
    print(f"  Sources    : {len(result.source_nodes)} node(s) used")
    print()


if __name__ == "__main__":
    main()
