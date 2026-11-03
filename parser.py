"""
parser.py - Phase 1: Academic PDF Parser
=========================================
Converts research paper PDFs into clean, structured Markdown files
that preserve section headers, tables, lists, and paragraph flow
for downstream chunking by ingest.py.

Workflow:
    1. Scan ./data/raw_pdfs/ for PDF files
    2. Extract and structure content using PyMuPDF (fitz)
    3. Reconstruct academic document structure (title, abstract,
       sections, subsections, tables, references)
    4. Write one clean .md file per PDF to ./data/parsed_markdown/

Usage (standalone):
    python parser.py
    python parser.py --input ./data/raw_pdfs --output ./data/parsed_markdown
    python parser.py --reset    # overwrite existing .md files

Called by main.py Option 1 via:
    import parser; parser.main()
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# ── PDF extraction ────────────────────────────────────────────────────────────
try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")
    import fitz  # PyMuPDF
except ImportError:
    print(
        "[ERROR] PyMuPDF is not installed.\n"
        "  Run:  pip install pymupdf\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RAW_PDF_DIR   = Path("./data/raw_pdfs")
PARSED_MD_DIR = Path("./data/parsed_markdown")

# Font-size thresholds for heading detection (points)
# Calibrated for typical academic paper layouts
TITLE_FONT_SIZE_MIN    = 14.0
HEADING1_FONT_SIZE_MIN = 12.0
HEADING2_FONT_SIZE_MIN = 10.5
BODY_FONT_SIZE_MIN     =  8.0   # ignore headers/footers below this

# Known academic section keywords for heuristic header detection
SECTION_KEYWORDS = {
    "abstract", "introduction", "background", "related work",
    "literature review", "methodology", "method", "methods",
    "approach", "proposed", "framework", "architecture",
    "experiment", "experiments", "experimental", "evaluation",
    "results", "discussion", "conclusion", "conclusions",
    "future work", "references", "bibliography", "appendix",
    "acknowledgment", "acknowledgements", "overview", "analysis",
    "implementation", "dataset", "datasets", "baseline", "baselines",
    "comparison", "limitations", "contributions",
}

WIDTH = 70


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'=' * WIDTH}")
    print(f"  {title}")
    print(f"{'=' * WIDTH}")


def _info(msg: str) -> None:
    print(f"  • {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}", file=sys.stderr)


def _error(msg: str) -> None:
    print(f"  [ERROR] {msg}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning utilities
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Clean raw extracted text:
    - Normalise unicode hyphens and dashes to ASCII
    - Remove soft hyphens used for PDF line-break hyphenation
    - Collapse excessive whitespace
    - Strip stray control characters
    """
    # Unicode normalisation for common academic text artifacts
    replacements = {
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u00ad": "",    # soft hyphen
        "\u00a0": " ",   # non-breaking space
        "\ufb01": "fi",  # fi ligature
        "\ufb02": "fl",  # fl ligature
        "\u2022": "-",   # bullet
        "\u2212": "-",   # minus sign
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse runs of spaces (but preserve newlines)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def _is_likely_header(text: str, font_size: float, avg_body_size: float) -> tuple[bool, int]:
    """
    Heuristically determine if a text span is a section header.

    Returns (is_header, level) where level is 1, 2, or 3.
    Uses a combination of:
    - Font size relative to body text
    - Presence of academic section keywords
    - Numbering patterns (e.g. "1.", "2.1", "III.")
    - Text length (headers are usually short)
    """
    text_stripped = text.strip()
    text_lower    = text_stripped.lower().rstrip(".")

    if not text_stripped or len(text_stripped) > 120:
        return False, 0

    # Numbered section pattern: "1.", "2.1", "1.2.3", "I.", "II.", "III."
    numbered = bool(re.match(
        r"^(\d+\.)+\s*\S|^[IVXLCDM]+\.\s*\S",
        text_stripped,
        re.IGNORECASE,
    ))

    # Keyword match
    keyword_match = any(
        text_lower == kw or text_lower.startswith(kw + " ") or text_lower.endswith(" " + kw)
        for kw in SECTION_KEYWORDS
    )

    size_ratio = font_size / avg_body_size if avg_body_size > 0 else 1.0

    # Determine level
    if font_size >= TITLE_FONT_SIZE_MIN and (numbered or keyword_match or size_ratio >= 1.4):
        return True, 1
    if font_size >= HEADING1_FONT_SIZE_MIN and (numbered or keyword_match or size_ratio >= 1.2):
        return True, 1
    if font_size >= HEADING2_FONT_SIZE_MIN and (numbered or keyword_match or size_ratio >= 1.1):
        return True, 2
    if (numbered or keyword_match) and size_ratio >= 1.0:
        return True, 3

    return False, 0


def _is_page_artifact(text: str, page_number: int) -> bool:
    """
    Detect and discard common PDF page artifacts:
    - Page numbers (standalone digits)
    - Running headers/footers (very short repeated lines)
    - DOI / URL lines
    - Copyright notices
    """
    t = text.strip()
    if not t:
        return True
    if re.match(r"^\d{1,4}$", t):          # bare page number
        return True
    if re.match(r"^https?://\S+$", t):     # bare URL
        return True
    if len(t) < 4:                          # too short to be content
        return True
    if re.match(r"^doi:", t, re.IGNORECASE):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Table detection & formatting
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """
    Detect table-like structures on a page using PyMuPDF's drawing analysis.

    Returns a list of (bounding_rect, markdown_table_string) tuples.
    Each tuple marks a region that has been converted to Markdown so the
    main text extractor can skip it and avoid double-extraction.

    Strategy:
    - Find rectangular drawing paths (table borders)
    - Extract text within those regions
    - Format as a Markdown pipe table
    """
    tables: list[tuple[fitz.Rect, str]] = []

    try:
        # Get all drawing paths on the page
        paths = page.get_drawings()
        if not paths:
            return tables

        # Find rects that look like table cells (small, numerous, aligned)
        rects: list[fitz.Rect] = []
        for path in paths:
            r = path.get("rect")
            if r and r.width > 20 and r.height > 5:
                rects.append(fitz.Rect(r))

        if len(rects) < 4:  # need at least a 2x2 grid
            return tables

        # Group rects by approximate Y position (rows)
        rects_sorted = sorted(rects, key=lambda r: (round(r.y0 / 5) * 5, r.x0))
        rows: dict[int, list[fitz.Rect]] = {}
        for rect in rects_sorted:
            row_key = round(rect.y0 / 8) * 8
            rows.setdefault(row_key, []).append(rect)

        # Only process if we have at least 2 rows with 2+ cells each
        valid_rows = {k: v for k, v in rows.items() if len(v) >= 2}
        if len(valid_rows) < 2:
            return tables

        # Build a bounding box for the entire table
        all_rects = [r for row in valid_rows.values() for r in row]
        table_bbox = fitz.Rect(
            min(r.x0 for r in all_rects),
            min(r.y0 for r in all_rects),
            max(r.x1 for r in all_rects),
            max(r.y1 for r in all_rects),
        )

        # Extract text from the table region
        table_text = page.get_text("blocks", clip=table_bbox)
        if not table_text:
            return tables

        # Organise text blocks into rows by Y coordinate
        text_rows: dict[int, list[tuple[float, str]]] = {}
        for block in table_text:
            if block[6] != 0:  # skip image blocks
                continue
            bx0, by0, bx1, by1, text, *_ = block
            row_key = round(by0 / 8) * 8
            cleaned = _clean_text(text.replace("\n", " ").strip())
            if cleaned:
                text_rows.setdefault(row_key, []).append((bx0, cleaned))

        if len(text_rows) < 2:
            return tables

        # Sort each row's cells by X position and build Markdown table
        md_rows: list[list[str]] = []
        for row_key in sorted(text_rows.keys()):
            cells = [cell for _, cell in sorted(text_rows[row_key])]
            md_rows.append(cells)

        if not md_rows:
            return tables

        # Normalise column count
        max_cols = max(len(r) for r in md_rows)
        for row in md_rows:
            while len(row) < max_cols:
                row.append("")

        # Build pipe-table Markdown
        lines: list[str] = []
        lines.append("| " + " | ".join(md_rows[0]) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in md_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        tables.append((table_bbox, "\n".join(lines)))

    except Exception:
        pass  # table extraction is best-effort; never crash the parser

    return tables


# ─────────────────────────────────────────────────────────────────────────────
# Core PDF → Markdown converter
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_markdown(pdf_path: Path) -> str:
    """
    Convert a single academic PDF to structured Markdown.

    Extraction strategy
    -------------------
    1. Use PyMuPDF's block-level text extraction which groups text by
       approximate visual line/paragraph — better than raw character soup.
    2. Compute the median body font size from all text spans to calibrate
       heading detection thresholds dynamically per document.
    3. For each block, classify it as: title, heading (H1/H2/H3),
       list item, table (from drawing analysis), or body paragraph.
    4. Reconstruct document flow in reading order (top-to-bottom,
       left-to-right, handling two-column layouts).
    5. Apply light post-processing: merge hyphenated line breaks,
       remove duplicate whitespace, ensure paragraph spacing.

    Returns the full Markdown string.
    """
    doc = fitz.open(str(pdf_path))

    # ── Pass 1: Compute median body font size across all pages ────────────────
    all_sizes: list[float] = []
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if BODY_FONT_SIZE_MIN <= size <= 20:
                        all_sizes.append(size)

    if all_sizes:
        all_sizes.sort()
        avg_body_size = all_sizes[len(all_sizes) // 2]  # median
    else:
        avg_body_size = 10.0

    # ── Pass 2: Extract and structure content page by page ────────────────────
    md_parts: list[str] = []
    seen_texts: set[str] = set()   # deduplicate running headers/footers
    first_heading_seen = False
    current_paragraph_lines: list[str] = []

    def _flush_paragraph() -> None:
        """Emit accumulated paragraph lines as a Markdown paragraph."""
        if current_paragraph_lines:
            para = " ".join(current_paragraph_lines).strip()
            para = re.sub(r"(\w)-\s+(\w)", r"\1\2", para)  # rejoin hyphenated words
            if para:
                md_parts.append(para + "\n")
            current_paragraph_lines.clear()

    for page_num, page in enumerate(doc):
        page_width  = page.rect.width
        page_height = page.rect.height

        # Detect table regions to avoid double-extracting their text
        table_regions = _extract_tables(page)
        table_bboxes  = [t[0] for t in table_regions]

        # Emit any detected tables
        for _, table_md in table_regions:
            _flush_paragraph()
            md_parts.append("\n" + table_md + "\n")

        # Extract text blocks in reading order
        blocks = page.get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_DEHYPHENATE,
            sort=True,   # reading order sort
        )["blocks"]

        for block in blocks:
            if block.get("type") != 0:   # skip image blocks
                continue

            # Skip blocks inside detected table regions
            block_rect = fitz.Rect(block["bbox"])
            in_table = any(
                block_rect.intersects(tb) and
                (block_rect & tb).get_area() > block_rect.get_area() * 0.5
                for tb in table_bboxes
            )
            if in_table:
                continue

            for line in block.get("lines", []):
                # Merge all spans in this line into one text + dominant font size
                # Insert a space between spans when the previous span doesn't
                # end with a space and next doesn't start with one (fixes
                # compressed academic PDFs that embed fonts without spacing).
                line_text  = ""
                max_size   = 0.0
                is_bold    = False

                for span in line.get("spans", []):
                    span_text = _clean_text(span.get("text", ""))
                    span_size = span.get("size", 0)
                    span_font = span.get("font", "").lower()

                    if span_size > max_size:
                        max_size = span_size
                    if "bold" in span_font or span.get("flags", 0) & 2**4:
                        is_bold = True

                    # Ensure a space boundary between adjacent spans
                    if line_text and span_text:
                        if not line_text.endswith(" ") and not span_text.startswith(" "):
                            line_text += " "
                    line_text += span_text

                # Fix run-together words caused by missing spaces in font encoding.
                # Insert space before uppercase letters that follow lowercase
                # only when there's no space already (conservative fix).
                line_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", line_text)
                # Fix missing space after punctuation followed immediately by a letter
                line_text = re.sub(r"([,;:\.\?!])([A-Za-z])", r"\1 \2", line_text)
                line_text = line_text.strip()

                if not line_text:
                    continue
                if max_size < BODY_FONT_SIZE_MIN:
                    continue
                if _is_page_artifact(line_text, page_num + 1):
                    continue

                # Deduplicate running headers/footers
                text_key = line_text.lower()[:60]
                if text_key in seen_texts and len(line_text) < 80:
                    continue
                seen_texts.add(text_key)

                # ── Classify this line ────────────────────────────────────────
                is_header, level = _is_header_line(
                    line_text, max_size, avg_body_size, is_bold
                )

                if is_header:
                    _flush_paragraph()
                    prefix = "#" * level
                    md_parts.append(f"\n{prefix} {line_text}\n")
                    first_heading_seen = True

                elif _is_list_item(line_text):
                    _flush_paragraph()
                    # Normalise various bullet styles to Markdown "-"
                    item_text = re.sub(r"^[\u2022\u2023\u25e6\u2043\*\-\+]\s*", "", line_text)
                    md_parts.append(f"- {item_text.strip()}\n")

                else:
                    # Body text — accumulate into paragraph
                    # Detect paragraph break: significant vertical gap or
                    # block boundary (handled implicitly by block iteration)
                    current_paragraph_lines.append(line_text)

            # End of block → flush paragraph
            _flush_paragraph()
            md_parts.append("")   # blank line between blocks

    _flush_paragraph()
    doc.close()

    # ── Post-processing ───────────────────────────────────────────────────────
    markdown = "\n".join(md_parts)

    # Collapse 3+ consecutive blank lines to 2
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    # Add document metadata header
    header = (
        f"# {pdf_path.stem.replace('_', ' ').replace('-', ' ')}\n\n"
        f"> Source: `{pdf_path.name}`\n\n"
        "---\n\n"
    )

    return header + markdown.strip() + "\n"


def _is_header_line(
    text: str,
    font_size: float,
    avg_body_size: float,
    is_bold: bool,
) -> tuple[bool, int]:
    """
    Wrapper around _is_likely_header that also considers bold formatting.
    Bold text at body size is treated as H3 if it matches a section keyword.
    """
    is_hdr, level = _is_likely_header(text, font_size, avg_body_size)
    if is_hdr:
        return True, level

    # Bold body-size text that matches section keywords → H3
    if is_bold and avg_body_size > 0:
        text_lower = text.strip().lower().rstrip(".")
        if any(text_lower == kw or text_lower.startswith(kw) for kw in SECTION_KEYWORDS):
            if len(text.strip()) < 80:
                return True, 3

    return False, 0


def _is_list_item(text: str) -> bool:
    """
    Detect lines that are list items based on leading bullet or number pattern.
    """
    return bool(re.match(
        r"^[\u2022\u2023\u25e6\u2043\*\-\+]\s+\S|"   # bullet characters
        r"^\(\d+\)\s+\S|"                              # (1) style
        r"^\d+[\.\)]\s+\S",                            # 1. or 1) style
        text.strip(),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Directory-level processing
# ─────────────────────────────────────────────────────────────────────────────

def parse_all_pdfs(
    input_dir:  Path = RAW_PDF_DIR,
    output_dir: Path = PARSED_MD_DIR,
    reset:      bool = False,
) -> list[Path]:
    """
    Scan *input_dir* for PDF files, convert each to Markdown, and write
    the results to *output_dir*.

    Parameters
    ----------
    input_dir  : Directory containing raw PDF files.
    output_dir : Directory to write .md output files.
    reset      : If True, overwrite existing .md files. If False, skip
                 PDFs that already have a corresponding .md file.

    Returns
    -------
    List of Path objects for each successfully written .md file.
    """
    _banner("Phase 1 — Academic PDF Parser")

    if not input_dir.exists():
        _error(f"Input directory not found: {input_dir.resolve()}")
        _error("Create the directory and add PDF files, then retry.")
        return []

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        _warn(f"No PDF files found in: {input_dir.resolve()}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    _info(f"Input  : {input_dir.resolve()}")
    _info(f"Output : {output_dir.resolve()}")
    _info(f"Found  : {len(pdf_files)} PDF file(s)")
    _info(f"Mode   : {'overwrite (reset)' if reset else 'incremental (skip existing)'}")
    print()

    written: list[Path] = []
    skipped = 0
    failed  = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        out_path = output_dir / (pdf_path.stem + ".md")

        # Skip if already parsed and not in reset mode
        if out_path.exists() and not reset:
            _info(f"[{i}/{len(pdf_files)}] SKIP  {pdf_path.name}  (already parsed)")
            skipped += 1
            continue

        print(f"  [{i}/{len(pdf_files)}] Parsing: {pdf_path.name} …", end="", flush=True)
        t0 = time.perf_counter()

        try:
            markdown = pdf_to_markdown(pdf_path)
            out_path.write_text(markdown, encoding="utf-8")
            elapsed = time.perf_counter() - t0
            char_count = len(markdown)
            print(f"  done  ({char_count:,} chars, {elapsed:.1f}s)")
            written.append(out_path)

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  FAILED ({elapsed:.1f}s)")
            _error(f"Could not parse {pdf_path.name}:\n    {exc}")
            failed += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    _ok(f"Parsing complete: {len(written)} parsed, {skipped} skipped, {failed} failed.")

    if written:
        print()
        print("  Output files:")
        for md_path in written:
            size_kb = md_path.stat().st_size / 1024
            print(f"    • {md_path.name}  ({size_kb:.1f} KB)")

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Public API (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def main(
    input_dir:  Path = RAW_PDF_DIR,
    output_dir: Path = PARSED_MD_DIR,
    reset:      bool = False,
) -> list[Path]:
    """
    Entry point called by main.py via importlib.

    Returns the list of successfully written .md file paths.
    """
    return parse_all_pdfs(
        input_dir=input_dir,
        output_dir=output_dir,
        reset=reset,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (standalone use)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="parser.py",
        description="Phase 1: Convert academic PDF papers to structured Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python parser.py\n"
            "  python parser.py --reset\n"
            "  python parser.py --input ./papers --output ./markdown\n"
        ),
    )
    parser.add_argument(
        "--input", type=Path, default=RAW_PDF_DIR,
        help=f"Directory containing PDF files  (default: {RAW_PDF_DIR})"
    )
    parser.add_argument(
        "--output", type=Path, default=PARSED_MD_DIR,
        help=f"Directory for .md output files  (default: {PARSED_MD_DIR})"
    )
    parser.add_argument(
        "--reset", action="store_true", default=False,
        help="Overwrite existing .md files (default: skip already-parsed PDFs)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = parse_all_pdfs(
        input_dir=args.input,
        output_dir=args.output,
        reset=args.reset,
    )
    sys.exit(0 if results is not None else 1)
