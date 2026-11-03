"""
main.py — Phase 4: Interactive CLI Orchestrator
================================================
Ties together all phases of the local academic RAG system into a single,
user-friendly terminal interface.

    Phase 1  parser.py   — PDF  →  Markdown
    Phase 2  ingest.py   — Markdown  →  ChromaDB vectors
    Phase 3  query.py    — Question  →  Cited answer (Ollama + hybrid search)

Usage:
    python main.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project paths  (mirrors ingest.py / query.py defaults)
# ─────────────────────────────────────────────────────────────────────────────

PARSED_MD_DIR   = Path("./data/parsed_markdown")
CHROMA_DB_DIR   = Path("./data/chroma_db")
COLLECTION_NAME = "academic_rag"

WIDTH = 70


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'=' * WIDTH}")
    print(f"  {title}")
    print(f"{'=' * WIDTH}")


def _section(title: str) -> None:
    print(f"\n{'-' * WIDTH}")
    print(f"  {title}")
    print(f"{'-' * WIDTH}")


def _info(msg: str) -> None:
    print(f"  • {msg}")


def _warn(msg: str) -> None:
    print(f"\n  ⚠  WARNING: {msg}")


def _error(msg: str) -> None:
    print(f"\n  [ERROR] {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _pause() -> None:
    """Wait for the user to press Enter before returning to the main menu."""
    print()
    input("  Press Enter to return to the main menu…")


# ─────────────────────────────────────────────────────────────────────────────
# Startup checks
# ─────────────────────────────────────────────────────────────────────────────

def check_data_directories() -> None:
    """
    Verify required data folders exist and warn the user if they are missing.
    Folders are created automatically where safe to do so.
    """
    created: list[str] = []
    missing_warnings: list[str] = []

    # parsed_markdown — Phase 1 output; we can create it but can't fill it
    if not PARSED_MD_DIR.exists():
        PARSED_MD_DIR.mkdir(parents=True, exist_ok=True)
        created.append(str(PARSED_MD_DIR))
        missing_warnings.append(
            f"'{PARSED_MD_DIR}' was just created but is empty.\n"
            "    Place your parsed .md files (or raw PDFs for Phase 1) there\n"
            "    before running Option 1."
        )

    # chroma_db — auto-created by ingest.py; we just ensure the parent exists
    if not CHROMA_DB_DIR.exists():
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        created.append(str(CHROMA_DB_DIR))

    if created:
        for path in created:
            _info(f"Created missing directory: {path}")

    for warning in missing_warnings:
        _warn(warning)


def is_chroma_empty() -> bool:
    """
    Return True if the ChromaDB collection has zero vectors — meaning the
    user hasn't run ingestion yet and querying would fail immediately.
    """
    try:
        import chromadb
        if not CHROMA_DB_DIR.exists():
            return True
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return collection.count() == 0
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Option 1 — Parse and Index New Papers
# ─────────────────────────────────────────────────────────────────────────────

def run_parse_and_index() -> None:
    """
    Sequentially run Phase 1 (parser.py) then Phase 2 (ingest.py).

    parser.py is optional — if it hasn't been created yet, this step skips
    Phase 1 and proceeds directly to ingestion of any .md files already in
    the parsed_markdown directory.
    """
    _banner("Option 1 — Parse and Index New Papers")

    # ── Phase 1: PDF → Markdown ───────────────────────────────────────────────
    print("\n  [ Phase 1 — PDF Parsing ]")
    parser_path = Path("./parser.py")

    if not parser_path.exists():
        _warn(
            "parser.py not found in the project root.\n"
            "    Phase 1 (PDF → Markdown) will be skipped.\n"
            "    If you have pre-converted .md files in\n"
            f"    '{PARSED_MD_DIR}', ingestion will proceed normally."
        )
    else:
        try:
            _info("Importing parser module …")
            # Import the parser module dynamically so the CLI doesn't hard-fail
            # when parser.py doesn't exist yet.
            import importlib.util
            spec = importlib.util.spec_from_file_location("parser", parser_path)
            parser_mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
            spec.loader.exec_module(parser_mod)                  # type: ignore[union-attr]

            # Call the module's main() or run() entry point
            if hasattr(parser_mod, "main"):
                _info("Running parser.main() …")
                parser_mod.main()
            elif hasattr(parser_mod, "run"):
                _info("Running parser.run() …")
                parser_mod.run()
            else:
                _warn(
                    "parser.py has no main() or run() function.\n"
                    "    Phase 1 skipped — proceeding to ingestion."
                )
        except Exception as exc:
            _error(f"Phase 1 failed:\n    {exc}")
            print("  Continuing to Phase 2 with any existing .md files …")

    # ── Phase 2: Markdown → ChromaDB ─────────────────────────────────────────
    print("\n  [ Phase 2 — Chunking & Indexing ]")

    md_files = list(PARSED_MD_DIR.glob("*.md"))
    if not md_files:
        _warn(
            f"No .md files found in '{PARSED_MD_DIR}'.\n"
            "    Add parsed Markdown files (or run Phase 1) and try again."
        )
        _pause()
        return

    _info(f"Found {len(md_files)} .md file(s) to index.")

    try:
        from ingest import (
            load_documents,
            chunk_documents,
            init_chroma,
            init_embedding_model,
            build_and_persist_index,
            print_final_report,
            PARSED_MD_DIR as INGEST_MD_DIR,
            CHROMA_DB_DIR as INGEST_DB_DIR,
            COLLECTION_NAME as INGEST_COLLECTION,
        )

        documents = load_documents(PARSED_MD_DIR)
        nodes = chunk_documents(documents)
        collection, vector_store = init_chroma(
            chroma_dir=CHROMA_DB_DIR,
            collection_name=COLLECTION_NAME,
            reset=False,          # incremental append; use ingest.py --reset for full wipe
        )
        init_embedding_model(model_key="bge")
        build_and_persist_index(nodes=nodes, vector_store=vector_store)
        print_final_report(
            collection=collection,
            nodes=nodes,
            chroma_dir=CHROMA_DB_DIR,
            model_key="bge",
        )
        _ok("Indexing pipeline completed successfully.")

    except SystemExit:
        # ingest.py calls sys.exit() on fatal errors — intercept and surface
        _error("Ingestion pipeline exited with an error (see output above).")
    except Exception as exc:
        _error(f"Ingestion pipeline failed:\n    {exc}")

    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# Option 2 — Ask a Question
# ─────────────────────────────────────────────────────────────────────────────

def run_ask_question() -> None:
    """
    Prompt the user for a query, run hybrid RAG retrieval + LLM synthesis,
    and print the cited answer alongside the retrieved source passages.
    """
    _banner("Option 2 — Ask a Question")

    # ── Safeguard: empty database ─────────────────────────────────────────────
    if is_chroma_empty():
        _warn(
            "The ChromaDB index is empty — no documents have been ingested yet.\n"
            "    Run Option 1 first to parse and index your academic papers."
        )
        _pause()
        return

    # ── Get query from user ───────────────────────────────────────────────────
    print()
    print("  Enter your research question below.")
    print("  (Leave blank and press Enter to cancel.)\n")
    try:
        user_query = input("  Question: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        _pause()
        return

    if not user_query:
        _info("No question entered. Returning to menu.")
        _pause()
        return

    # ── Run query ─────────────────────────────────────────────────────────────
    print()
    _info("Initialising query engine …")

    try:
        from query import run_academic_query

        result = run_academic_query(user_query)

    except FileNotFoundError as exc:
        _error(f"Index not found:\n    {exc}")
        _pause()
        return
    except RuntimeError as exc:
        _error(str(exc))
        _pause()
        return
    except ConnectionError as exc:
        _error(
            f"Cannot reach the Ollama LLM server.\n"
            f"    Make sure Ollama is running:  ollama serve\n"
            f"    Detail: {exc}"
        )
        _pause()
        return
    except KeyboardInterrupt:
        print("\n  Query interrupted.")
        _pause()
        return
    except Exception as exc:
        _error(f"Unexpected error during query:\n    {exc}")
        _pause()
        return

    # ── Print answer ──────────────────────────────────────────────────────────
    _section("Answer")
    # Wrap long lines at 86 chars for comfortable terminal reading
    for paragraph in result.response.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph, width=86, initial_indent="  ",
                                subsequent_indent="  "))
        else:
            print()

    # ── Print source passages ─────────────────────────────────────────────────
    _section(f"Retrieved Source Passages  ({len(result.source_nodes)} used)")
    if not result.source_nodes:
        _warn("No source nodes were returned above the similarity threshold.")
    else:
        for i, nws in enumerate(result.source_nodes, 1):
            meta    = nws.node.metadata
            fname   = Path(meta.get("file_name", "unknown")).name
            header  = meta.get("header_path", meta.get("section", "—"))
            score   = nws.score or 0.0
            snippet = nws.node.text[:200].replace("\n", " ").strip()
            print(
                f"\n  [{i}] {fname}\n"
                f"      Section : {header}\n"
                f"      Score   : {score:.4f}\n"
                f"      Preview : \"{snippet}…\""
            )

    # ── Summary line ─────────────────────────────────────────────────────────
    _section("Query Summary")
    _info(f"Elapsed   : {result.elapsed_sec:.1f}s")
    _info(f"Sources   : {len(result.source_nodes)} node(s) cited")

    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# Option 3 — Exit
# ─────────────────────────────────────────────────────────────────────────────

def run_exit() -> None:
    """
    Gracefully shut down: close any lingering ChromaDB client handles and exit.
    """
    _section("Shutting Down")

    # Close ChromaDB client if it was opened during the session
    try:
        import chromadb
        if CHROMA_DB_DIR.exists():
            client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            # PersistentClient persists on every write; explicit reset() flushes
            # the internal connection pool cleanly.
            client.reset()   # no-op if already clean; harmless to call
    except Exception:
        pass   # silently ignore cleanup errors on exit

    _ok("All connections closed.")
    print("\n  Goodbye!\n")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────────────────────────────────────

def print_menu() -> None:
    """Render the main menu."""
    print(f"\n{'=' * WIDTH}")
    print("  LOCAL ACADEMIC RAG SYSTEM")
    print(f"{'=' * WIDTH}")
    print("  1.  Parse and Index New Papers")
    print("  2.  Ask a Question")
    print("  3.  Exit")
    print(f"{'-' * WIDTH}")


def get_menu_choice() -> str:
    """Read a single menu choice from the user, returning it as a stripped string."""
    try:
        choice = input("  Your choice [1/2/3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C / Ctrl+D — treat as exit
        print()
        choice = "3"
    return choice


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("RAG System — Starting Up")

    # ── Startup sanity checks ─────────────────────────────────────────────────
    _info("Checking required directories …")
    check_data_directories()

    if is_chroma_empty():
        _warn(
            "The vector database is currently empty.\n"
            "    Select Option 1 to ingest your papers before querying."
        )

    _ok("System ready.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        print_menu()
        choice = get_menu_choice()

        if choice == "1":
            run_parse_and_index()
        elif choice == "2":
            run_ask_question()
        elif choice == "3":
            run_exit()
        else:
            print(f"\n  '{choice}' is not a valid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
