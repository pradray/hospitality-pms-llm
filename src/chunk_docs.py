"""Chunk PDF and Markdown documentation into sections for embedding.

Handles:
- PDFs with TOC bookmarks (oracle-guides, ohip-platform PDFs, supplementary)
- Markdown files with heading structure (OHIP user guide, implementation guides)

Each chunk gets metadata: module, doc_type, source_file, section_path.
"""

import json
import hashlib
import re
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent  # ../Dissertation
DATA = DATA_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "doc_chunks"

# --- Source registry ---
# (file_path relative to DATA, module, doc_type)

PDF_SOURCES = [
    ("oracle-guides/OPERA_Cloud_User_Guide_26.1.pdf", "multi", "user_guide"),
    ("oracle-guides/OPERA_Cloud_Release_Readiness_26.1.pdf", "multi", "release_notes"),
    ("ohip-platform/OHIP_Revenue_Credit_Management.pdf", "front_office", "impl_guide"),
    ("ohip-platform/OHIP_Housekeeping_Implementation.pdf", "housekeeping", "impl_guide"),
    ("ohip-platform/OHIP_ERP_Integration.pdf", "front_office", "impl_guide"),
    ("ohip-platform/OHIP_Datasheet.pdf", "multi", "overview"),
    ("supplementary/CRS_Integration_Guide.pdf", "reservations", "impl_guide"),
    ("supplementary/Meetings_Events_Integration_Guide.pdf", "reservations", "impl_guide"),
]

MD_SOURCES = [
    ("ohip-platform/user-guide-html/OHIP_User_Guide_26.1_Scraped.md", "multi", "user_guide"),
    ("ohip-platform/implementation-guides/01_Posting_Charges_Implementation_Guide.md", "front_office", "impl_guide"),
    ("ohip-platform/implementation-guides/02_Payment_Authorization_Settlement_Guide.md", "front_office", "impl_guide"),
    ("ohip-platform/implementation-guides/03_Housekeeping_Service_Status_Guide.md", "housekeeping", "impl_guide"),
    ("ohip-platform/implementation-guides/04_Guest_Messages_Guide.md", "front_office", "impl_guide"),
    ("ohip-platform/implementation-guides/05_Wake_Up_Calls_Guide.md", "front_office", "impl_guide"),
    ("ohip-platform/implementation-guides/06_Revenue_Management_System_Guide.md", "reservations", "impl_guide"),
    ("ohip-platform/implementation-guides/07_Streaming_Implementation_Guide.md", "multi", "impl_guide"),
    ("ohip-platform/implementation-guides/08_Distribution_Shop_and_Book_Guide.md", "reservations", "impl_guide"),
]

# Chapters in the User Guide relevant to our 4 modules
USER_GUIDE_CHAPTER_FILTER = {
    2: "crm",           # Client Relations / Profiles
    3: "reservations",   # Reservations
    4: "front_office",   # Front Desk
    5: "housekeeping",   # Inventory / Housekeeping (Room & Housekeeping Mgmt)
    6: "reservations",   # Blocks
    7: "front_office",   # Cashiering & Financials
    8: "front_office",   # Commission
    9: "front_office",   # End of Day
    10: "reservations",  # Rate Management
    # Chapters 11+ (Reports, Interfaces, Admin, etc.) — keep as "multi"
}

MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 4000
OVERLAP_CHARS = 200


def make_chunk_id(source: str, section: str) -> str:
    return hashlib.md5(f"{source}:{section}".encode()).hexdigest()[:12]


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 4 and not stripped[0].isalpha():
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


# --- PDF chunking ---

def get_chapter_number(title: str) -> int | None:
    m = re.match(r"^(\d+)\s", title.strip())
    return int(m.group(1)) if m else None


def chunk_pdf_by_toc(pdf_path: Path, module: str, doc_type: str) -> list[dict]:
    doc = pymupdf.open(str(pdf_path))
    toc = doc.get_toc()
    source_file = pdf_path.name
    chunks = []

    if not toc:
        # No TOC — chunk by page groups
        return chunk_pdf_by_pages(doc, source_file, module, doc_type)

    # Build sections from TOC: each entry = (level, title, start_page)
    # We chunk at L1/L2 granularity; L3+ gets merged into parent
    sections = []
    for i, (level, title, page) in enumerate(toc):
        if level > 2:
            continue
        title = title.strip()
        if title.lower() in ("contents", ""):
            continue

        # Determine end page
        end_page = doc.page_count
        for j in range(i + 1, len(toc)):
            next_level, _, next_page = toc[j]
            if next_level <= level:
                end_page = next_page
                break

        sections.append((level, title, page, end_page))

    # Extract text for each section
    for level, title, start_page, end_page in sections:
        # 0-indexed pages
        start_idx = max(0, start_page - 1)
        end_idx = min(doc.page_count, end_page - 1)

        text_parts = []
        for pg in range(start_idx, end_idx):
            page_text = doc[pg].get_text()
            if page_text:
                text_parts.append(page_text)

        full_text = clean_text("\n".join(text_parts))
        if len(full_text) < MIN_CHUNK_CHARS:
            continue

        # For the User Guide, assign module based on chapter number
        chunk_module = module
        if source_file.startswith("OPERA_Cloud_User_Guide"):
            ch_num = get_chapter_number(title)
            if ch_num and ch_num in USER_GUIDE_CHAPTER_FILTER:
                chunk_module = USER_GUIDE_CHAPTER_FILTER[ch_num]

        # Split oversized sections into sub-chunks
        sub_chunks = split_long_text(full_text, title)
        for idx, sub_text in enumerate(sub_chunks):
            section_label = title if len(sub_chunks) == 1 else f"{title} (part {idx+1})"
            chunks.append({
                "id": make_chunk_id(source_file, section_label),
                "module": chunk_module,
                "doc_type": doc_type,
                "source_file": source_file,
                "section": section_label,
                "level": level,
                "text": sub_text,
            })

    doc.close()
    return chunks


def chunk_pdf_by_pages(doc, source_file: str, module: str, doc_type: str) -> list[dict]:
    """Fallback for PDFs without TOC — chunk every 3 pages."""
    chunks = []
    pages_per_chunk = 3
    for start in range(0, doc.page_count, pages_per_chunk):
        end = min(start + pages_per_chunk, doc.page_count)
        text_parts = []
        for pg in range(start, end):
            text_parts.append(doc[pg].get_text())
        text = clean_text("\n".join(text_parts))
        if len(text) < MIN_CHUNK_CHARS:
            continue
        section = f"Pages {start+1}-{end}"
        for idx, sub_text in enumerate(split_long_text(text, section)):
            label = section if idx == 0 else f"{section} (part {idx+1})"
            chunks.append({
                "id": make_chunk_id(source_file, label),
                "module": module,
                "doc_type": doc_type,
                "source_file": source_file,
                "section": label,
                "level": 0,
                "text": sub_text,
            })
    return chunks


# --- Markdown chunking ---

def chunk_markdown(md_path: Path, module: str, doc_type: str) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")
    source_file = md_path.name
    chunks = []

    # Split on headings (# or ##)
    heading_pattern = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No headings — treat whole file as one chunk
        cleaned = clean_text(text)
        if len(cleaned) >= MIN_CHUNK_CHARS:
            for idx, sub in enumerate(split_long_text(cleaned, source_file)):
                label = source_file if idx == 0 else f"{source_file} (part {idx+1})"
                chunks.append({
                    "id": make_chunk_id(source_file, label),
                    "module": module,
                    "doc_type": doc_type,
                    "source_file": source_file,
                    "section": label,
                    "level": 1,
                    "text": sub,
                })
        return chunks

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = clean_text(text[start:end])

        if len(section_text) < MIN_CHUNK_CHARS:
            continue

        for idx, sub in enumerate(split_long_text(section_text, title)):
            label = title if idx == 0 else f"{title} (part {idx+1})"
            chunks.append({
                "id": make_chunk_id(source_file, label),
                "module": module,
                "doc_type": doc_type,
                "source_file": source_file,
                "section": label,
                "level": level,
                "text": sub,
            })

    return chunks


# --- Text splitting ---

def split_long_text(text: str, label: str) -> list[str]:
    """Split text exceeding MAX_CHUNK_CHARS into overlapping sub-chunks."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    sub_chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS

        # Try to break at paragraph boundary
        if end < len(text):
            para_break = text.rfind("\n\n", start + MAX_CHUNK_CHARS // 2, end)
            if para_break > start:
                end = para_break

        sub_chunks.append(text[start:end].strip())
        start = end - OVERLAP_CHARS

    return sub_chunks


# --- Main ---

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    print("=== PDFs ===")
    for rel_path, module, doc_type in PDF_SOURCES:
        pdf_path = DATA / rel_path
        if not pdf_path.exists():
            print(f"  SKIP (not found): {rel_path}")
            continue
        chunks = chunk_pdf_by_toc(pdf_path, module, doc_type)
        all_chunks.extend(chunks)
        print(f"  {pdf_path.name:50s} → {len(chunks):4d} chunks")

    print("\n=== Markdown ===")
    for rel_path, module, doc_type in MD_SOURCES:
        md_path = DATA / rel_path
        if not md_path.exists():
            print(f"  SKIP (not found): {rel_path}")
            continue
        chunks = chunk_markdown(md_path, module, doc_type)
        all_chunks.extend(chunks)
        print(f"  {md_path.name:50s} → {len(chunks):4d} chunks")

    # Write output
    output_path = OUTPUT_DIR / "all_doc_chunks.jsonl"
    with open(output_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks → {output_path}")

    # Stats
    by_module = {}
    by_doc_type = {}
    text_lengths = []
    for c in all_chunks:
        by_module[c["module"]] = by_module.get(c["module"], 0) + 1
        by_doc_type[c["doc_type"]] = by_doc_type.get(c["doc_type"], 0) + 1
        text_lengths.append(len(c["text"]))

    print("\nBy module:")
    for m, count in sorted(by_module.items()):
        print(f"  {m:20s} {count:4d}")
    print("\nBy doc_type:")
    for d, count in sorted(by_doc_type.items()):
        print(f"  {d:20s} {count:4d}")
    print(f"\nText length: min={min(text_lengths)}, median={sorted(text_lengths)[len(text_lengths)//2]}, max={max(text_lengths)}, mean={sum(text_lengths)//len(text_lengths)}")


if __name__ == "__main__":
    main()
