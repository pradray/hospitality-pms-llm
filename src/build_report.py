"""Build the dissertation DOCX and PDF to the BITS WILP formatting guidelines.

The guidelines require several things pandoc will not do on its own:

  * quarto page (9 x 11 in), 1 inch margins, double spacing
  * serial page numbers in the footer
  * lower-case roman numerals for the front matter, Arabic restarting at 1 on
    Chapter 1 ("Ch. 1 should start on Page # 1")
  * a table of contents whose page numbers are actually right

The TOC is the awkward one: a Word TOC field is only populated when Word or
LibreOffice updates it, which does not happen during a headless PDF conversion,
so the submitted PDF would ship with an empty or stale contents page. This
script therefore builds twice — once to discover where each heading lands, then
again with those page numbers written into the contents table.

Usage:
    python src/build_report.py
"""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DOCS = PROJECT_ROOT / "docs"
SRC_MD = DOCS / "final_report.md"
BASENAME = "2024CT05003_Final_Dissertation"
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Quarto page with 1-inch margins, per the guidelines.
PAGE_XML = ('<w:pgSz w:w="12960" w:h="15840"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="720" w:footer="720" w:gutter="0"/>')


def make_reference_docx(dest: Path):
    """pandoc's default reference doc, restyled to the guidelines."""
    base = Path("/tmp/_ref_default.docx")
    with open(base, "wb") as f:
        f.write(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                               capture_output=True).stdout)
    work = Path("/tmp/_refdx")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    with zipfile.ZipFile(base) as z:
        z.extractall(work)

    doc = work / "word" / "document.xml"
    s = doc.read_text()
    # quarto 9" x 11" = 12960 x 15840 twips; 1" margins = 1440 twips.
    # pandoc's default reference has NO pgSz/pgMar in its sectPr, so these must be
    # inserted, not substituted - a plain re.sub silently leaves the page at Letter.
    s = re.sub(r"<w:pgSz[^>]*/>", "", s)
    s = re.sub(r"<w:pgMar[^>]*/>", "", s)
    s = s.replace("</w:sectPr>", PAGE_XML + "</w:sectPr>")
    doc.write_text(s)

    styles = work / "word" / "styles.xml"
    t = styles.read_text()
    # The guidelines specify no typeface. The default reference doc uses Aptos,
    # which ships only with recent Office and has no metric-compatible substitute,
    # so the document is pinned to Times New Roman for body text and headings -
    # the convention in the BITS sample report - at 12pt.
    t = re.sub(r'w:ascii="[^"]*"', 'w:ascii="Times New Roman"', t)
    t = re.sub(r'w:hAnsi="[^"]*"', 'w:hAnsi="Times New Roman"', t)
    t = re.sub(r'w:cs="[^"]*"', 'w:cs="Times New Roman"', t)
    t = re.sub(r'w:eastAsia="[^"]*"', 'w:eastAsia="Times New Roman"', t)
    t = re.sub(r'<w:sz w:val="\d+"/>', '<w:sz w:val="24"/>', t)
    t = re.sub(r'<w:szCs w:val="\d+"/>', '<w:szCs w:val="24"/>', t)
    for sid in ("Normal", "BodyText", "FirstParagraph", "Compact"):
        pat = re.compile(r'(<w:style [^>]*w:styleId="%s".*?</w:style>)' % sid, re.S)
        m = pat.search(t)
        if not m:
            continue
        blk = m.group(1)
        spacing = '<w:spacing w:after="0" w:line="480" w:lineRule="auto"/>'
        if "<w:spacing" in blk:
            new = re.sub(r"<w:spacing[^/]*/>", spacing, blk)
        elif "<w:pPr>" in blk:
            new = blk.replace("<w:pPr>", "<w:pPr>" + spacing, 1)
        else:
            new = blk.replace("</w:style>", f"<w:pPr>{spacing}</w:pPr></w:style>")
        t = t.replace(blk, new)
    styles.write_text(t)

    # Headings reference *theme* fonts (w:asciiTheme="majorHAnsi"), which resolve
    # through theme1.xml, so editing styles.xml alone leaves Aptos in the headings.
    for theme in (work / "word" / "theme").glob("theme*.xml"):
        th = theme.read_text()
        th = re.sub(r'(<a:(?:majorFont|minorFont)>\s*<a:latin[^>]*typeface=")[^"]*"',
                    r'\1Times New Roman"', th)
        theme.write_text(th)

    if dest.exists():
        dest.unlink()
    subprocess.run(["zip", "-Xqr", str(dest), "."], cwd=work, check=True)


FOOTER_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{w}"><w:p><w:pPr><w:jc w:val="center"/>
<w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>
<w:r><w:fldChar w:fldCharType="begin"/></w:r>
<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>
<w:r><w:fldChar w:fldCharType="separate"/></w:r>
<w:r><w:t>1</w:t></w:r>
<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>"""


def add_page_numbering(docx_path: Path):
    """Split into two sections: roman front matter, Arabic body from Chapter 1."""
    work = Path("/tmp/_bookdx")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    with zipfile.ZipFile(docx_path) as z:
        z.extractall(work)

    # footer parts
    (work / "word" / "footer2.xml").write_text(FOOTER_XML.format(w=W))
    (work / "word" / "footer3.xml").write_text(FOOTER_XML.format(w=W))

    rels = work / "word" / "_rels" / "document.xml.rels"
    r = rels.read_text()
    add = ('<Relationship Id="rIdFtrA" Type="http://schemas.openxmlformats.org/'
           'officeDocument/2006/relationships/footer" Target="footer2.xml"/>'
           '<Relationship Id="rIdFtrB" Type="http://schemas.openxmlformats.org/'
           'officeDocument/2006/relationships/footer" Target="footer3.xml"/>')
    rels.write_text(r.replace("</Relationships>", add + "</Relationships>"))

    ct = work / "[Content_Types].xml"
    c = ct.read_text()
    ov = ('<Override PartName="/word/footer2.xml" ContentType="application/vnd.'
          'openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
          '<Override PartName="/word/footer3.xml" ContentType="application/vnd.'
          'openxmlformats-officedocument.wordprocessingml.footer+xml"/>')
    ct.write_text(c.replace("</Types>", ov + "</Types>"))

    doc = work / "word" / "document.xml"
    s = doc.read_text()

    # Single-space paragraphs inside tables. The double spacing the guidelines ask
    # for applies to body text; applied to table cells it wraps every cell over two
    # or three lines and makes the results tables hard to read.
    def single_space_tables(xml: str) -> str:
        out, pos = [], 0
        for m in re.finditer(r"<w:tbl>.*?</w:tbl>", xml, re.S):
            out.append(xml[pos:m.start()])
            tbl = m.group(0)
            tbl = re.sub(r'<w:spacing w:after="0" w:line="480" w:lineRule="auto"/>',
                         '<w:spacing w:after="20" w:line="240" w:lineRule="auto"/>', tbl)
            # cells whose paragraphs carry no explicit spacing inherit the double
            # spacing from Normal, so give them one
            tbl = re.sub(r"<w:pPr>(?!<w:spacing)",
                         '<w:pPr><w:spacing w:after="20" w:line="240" w:lineRule="auto"/>',
                         tbl)
            out.append(tbl)
            pos = m.end()
        out.append(xml[pos:])
        return "".join(out)

    s = single_space_tables(s)

    final_sect = re.search(r"<w:sectPr[^>]*>.*?</w:sectPr>", s, re.S).group(0)
    if "<w:pgSz" not in final_sect:
        new_sect = final_sect.replace("</w:sectPr>", PAGE_XML + "</w:sectPr>")
        s = s.replace(final_sect, new_sect)
        final_sect = new_sect
    pgsz = re.search(r"<w:pgSz[^>]*/>", final_sect).group(0)
    pgmar = re.search(r"<w:pgMar[^>]*/>", final_sect).group(0)

    # Front-matter section: roman numerals, ends just before Chapter 1.
    # OOXML fixes the order of sectPr children: footerReference first, then
    # pgSz/pgMar, and pgNumType only AFTER pgMar. Out of order, renderers
    # silently drop the numbering format.
    front_sect = (f'<w:p><w:pPr><w:sectPr>'
                  f'<w:footerReference w:type="default" r:id="rIdFtrA"/>'
                  f'{pgsz}{pgmar}'
                  f'<w:pgNumType w:fmt="lowerRoman" w:start="1"/>'
                  f'</w:sectPr></w:pPr></w:p>')

    # Locate the Chapter 1 HEADING paragraph. Matching on the text alone finds the
    # table-of-contents row first, which would put the section break inside the
    # front matter and leave the whole document in roman numerals, so the match is
    # constrained to a paragraph carrying a Heading style.
    target = None
    for m in re.finditer(r"<w:p\b.*?</w:p>", s, re.S):
        blk = m.group(0)
        if "1. Introduction" in blk and re.search(r'w:pStyle w:val="Heading[12]"', blk):
            target = m
            break
    if target is None:
        raise SystemExit("could not locate the '1. Introduction' chapter heading")
    s = s[:target.start()] + front_sect + s[target.start():]

    # Body section: Arabic restarting at 1, with the same ordering constraint.
    new_final = final_sect.replace(
        "<w:pgSz", '<w:footerReference w:type="default" r:id="rIdFtrB"/><w:pgSz', 1)
    new_final = new_final.replace(
        pgmar, pgmar + '<w:pgNumType w:fmt="decimal" w:start="1"/>', 1)
    s = s.replace(final_sect, new_final)

    # the document root needs the r: namespace for footerReference
    if 'xmlns:r=' not in s.split(">", 2)[1]:
        s = s.replace("<w:document ", '<w:document xmlns:r="http://schemas.'
                      'openxmlformats.org/officeDocument/2006/relationships" ', 1)
    doc.write_text(s)

    if docx_path.exists():
        docx_path.unlink()
    subprocess.run(["zip", "-Xqr", str(docx_path), "."], cwd=work, check=True)


def to_pdf(docx_path: Path):
    subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf",
                    "--outdir", str(docx_path.parent), str(docx_path)],
                   capture_output=True, check=True)
    return docx_path.with_suffix(".pdf")


PAGEBREAK = ('\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n')


def preprocess(md_text: str) -> str:
    """Convert \\newpage markers into real docx page breaks.

    `\\newpage` is LaTeX; the docx writer silently drops it, which runs the
    certificate, abstract and contents together on shared pages."""
    out = []
    for line in md_text.split("\n"):
        out.append(PAGEBREAK if line.strip() == "\\newpage" else line)
    return "\n".join(out)


def build_once(md: Path, out_docx: Path, ref: Path):
    staged = Path("/tmp/_staged.md")
    staged.write_text(preprocess(md.read_text()))
    subprocess.run(["pandoc", str(staged), f"--reference-doc={ref}",
                    "--resource-path", str(DOCS), "-o", str(out_docx)], check=True)
    add_page_numbering(out_docx)
    return to_pdf(out_docx)


def _roman(n: int) -> str:
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, sym in vals:
        while n >= v:
            out += sym
            n -= v
    return out


def heading_pages(pdf: Path, headings):
    """Map each contents entry to the page number actually printed on it.

    Body entries are searched only from the Chapter 1 page onward: the contents
    table itself lists every heading, so a document-wide search matches the
    contents page and yields a negative offset. Front-matter entries fall back to
    a roman numeral taken from the physical page.
    """
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    pages = txt.split("\f")
    start = None
    for i, p in enumerate(pages):
        if re.search(r"^\s*1\.?\s+Introduction\s*$", p, re.M):
            start = i
            break
    if start is None:
        return {}

    # The contents table lists every heading, so its own pages must be excluded
    # or a front-matter entry resolves to wherever it is listed rather than to
    # where it actually appears.
    toc_pages = {i for i, p in enumerate(pages[:start])
                 if "TABLE OF CONTENTS" in p or re.search(r"^\s*Section\s+Page\s*$", p, re.M)}

    # A long heading wraps across lines in the rendered PDF, so an anchored
    # single-line match silently misses it and the placeholder page number
    # survives into the contents. Compare against whitespace-collapsed text too.
    flat = [re.sub(r"\s+", " ", p).lower() for p in pages]

    found = {}
    for h in headings:
        label = h.strip()
        probe = re.compile(r"^\s*" + re.escape(label) + r"\s*$", re.M | re.I)
        # Match at line start, trying progressively shorter prefixes. Anchoring at
        # the start rules out a short label such as "References" matching the same
        # word mid-sentence, while the shorter prefixes still find a heading that
        # the renderer wrapped over two lines.
        probes = [probe]
        for cut in (40, 30, 22):
            if len(label) > cut:
                probes.append(re.compile(r"^\s*" + re.escape(label[:cut]), re.M | re.I))
        probes.append(re.compile(r"^\s*" + re.escape(label) + r"\b", re.M | re.I))
        hit = None
        for i in range(start, len(pages)):          # body: Arabic
            if any(pr.search(pages[i]) for pr in probes):
                hit = str(i - start + 1)
                break
        if hit is None:                             # front matter: roman
            probe_ci = re.compile(r"^\s*" + re.escape(label) + r"\s*$", re.M | re.I)
            for i in range(0, start):
                if i in toc_pages:
                    continue
                if probe_ci.search(pages[i]):
                    hit = _roman(i + 1)
                    break
        if hit:
            found[label] = hit
    return found


def main():
    ref = Path("/tmp/reference_bits_final.docx")
    make_reference_docx(ref)
    print("reference docx styled: quarto page, 1in margins, double spacing")

    md = SRC_MD.read_text()

    # collect the headings listed in the contents table
    toc_rows = re.findall(r"^\| ([^|]+?) \| [ivx0-9]+ \|$", md, re.M)
    toc_rows = [t.strip() for t in toc_rows if t.strip() not in ("Section",)]

    tmp_md = Path("/tmp/_report_pass1.md")
    tmp_md.write_text(md)
    out_docx = DOCS / f"{BASENAME}.docx"
    pdf = build_once(tmp_md, out_docx, ref)
    print(f"pass 1 built: {pdf.name}")

    pages = heading_pages(pdf, toc_rows)
    print(f"located {len(pages)}/{len(toc_rows)} contents entries in the PDF")

    def fix(m):
        label = m.group(1).strip()
        if label in pages:
            return f"| {m.group(1)} | {pages[label]} |"
        return m.group(0)

    md2 = re.sub(r"^\| ([^|]+?) \| [ivx0-9]+ \|$", fix, md, flags=re.M)
    tmp2 = Path("/tmp/_report_pass2.md")
    tmp2.write_text(md2)
    pdf = build_once(tmp2, out_docx, ref)
    print(f"pass 2 built with corrected contents: {pdf.name}")

    size = pdf.stat().st_size
    n = len(subprocess.run(["pdftotext", str(pdf), "-"],
                           capture_output=True, text=True).stdout.split("\f"))
    print(f"\n{pdf.name}: {size/1024:.0f} KB, {n} pages")
    print(f"{out_docx.name}: {out_docx.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
