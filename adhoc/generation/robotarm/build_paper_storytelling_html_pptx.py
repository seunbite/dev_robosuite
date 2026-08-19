#!/usr/bin/env python3
"""Build HTML (+ PPTX) for PAPER_STORYTELLING_HYPOTHESES.md with embedded PNGs."""
from __future__ import annotations

import html as H
import re
import sys
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MD = REPO / "PAPER_STORYTELLING_HYPOTHESES.md"
OUT_HTML = REPO / "data/results/html/paper_storytelling_hypotheses.html"
OUT_PPTX = REPO / "data/results/evaluation/paper_storytelling_hypotheses.pptx"

PRODUCT = "LLM Knowledge to Robots Gesture Generation"
TITLE = "Can LLMs turn gesture knowledge into recognizable robot motion?"
ARC = "Yes, but approximately → Generation → Knowledge across embodiments → Identification across embodiments"


def _clean_path(path_str: str) -> str:
    clean = path_str.strip().strip("`").strip()
    clean = re.sub(r"\s*\(\+[^)]*\)\s*$", "", clean)
    clean = clean.strip().strip("`").split()[0].strip("`")
    return clean


def href_for(path_str: str) -> str | None:
    clean = _clean_path(path_str)
    p = (REPO / clean).resolve()
    if not p.is_file():
        print("MISSING", clean, file=sys.stderr)
        return None
    return "../" + str(p.relative_to(REPO / "data" / "results")).replace("\\", "/")


def inline_fmt(text: str) -> str:
    segs = re.split(r"(\*\*.+?\*\*|`[^`]+`)", text)
    frag: list[str] = []
    for part in segs:
        m = re.fullmatch(r"\*\*(.+)\*\*", part)
        if m:
            frag.append(f"<b>{H.escape(m.group(1))}</b>")
            continue
        m = re.fullmatch(r"`([^`]+)`", part)
        if m:
            frag.append(f"<code>{H.escape(m.group(1))}</code>")
            continue
        frag.append(H.escape(part))
    return "".join(frag)


def render_input_block(lines: list[str]) -> str:
    body_lines: list[str] = []
    for raw in lines:
        if not raw.strip():
            body_lines.append("")
            continue
        body_lines.append(inline_fmt(raw))
    inner = "<br>\n".join(body_lines)
    return f'<div class="input-example">{inner}</div>'


def render_table(table_buf: list[str]) -> str:
    header = [c.strip() for c in table_buf[0].strip("|").split("|")]
    body_lines = table_buf[2:] if len(table_buf) > 2 else []
    # Parse rows; "^" in a cell means rowspan continue from the previous non-^ value.
    rows: list[list[str]] = []
    for tl in body_lines:
        cells = [c.strip() for c in tl.strip("|").split("|")]
        while len(cells) < len(header):
            cells.append("")
        rows.append(cells[: len(header)])

    # Compute rowspan per cell: for each col, collapse runs of "^"
    rowspan = [[1] * len(header) for _ in rows]
    skip = [[False] * len(header) for _ in rows]
    for col in range(len(header)):
        i = 0
        while i < len(rows):
            if rows[i][col] == "^":
                # orphan ^ — treat as empty
                rows[i][col] = ""
                i += 1
                continue
            j = i + 1
            while j < len(rows) and rows[j][col] == "^":
                skip[j][col] = True
                j += 1
            rowspan[i][col] = j - i
            i = j

    text_headers = {
        "channel",
        "morphology",
        "platform",
        "embodiment",
        "group",
        "cue",
        "label",
        "bucket",
        "#1",
        "#2",
        "#3",
        "#4",
        "#5",
    }
    highlight_headers = {"accurate", "diff"}

    def col_class(name: str) -> str:
        key = name.strip().lower()
        classes: list[str] = []
        if key in text_headers:
            classes.append("col-text")
        else:
            classes.append("col-num")
        if key in highlight_headers:
            classes.append("col-hi")
        return " ".join(classes)

    ths = "".join(
        f'<th class="{col_class(c)}">{H.escape(c)}</th>' for c in header
    )
    parts = [f"<table><thead><tr>{ths}</tr></thead><tbody>"]
    for r, row in enumerate(rows):
        tds = []
        for c, val in enumerate(row):
            if skip[r][c]:
                continue
            rs = rowspan[r][c]
            attr = f' rowspan="{rs}"' if rs > 1 else ""
            cls = col_class(header[c])
            tds.append(f'<td class="{cls}"{attr}>{inline_fmt(val)}</td>')
        parts.append("<tr>" + "".join(tds) + "</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def md_to_html_body(text: str) -> str:
    out: list[str] = []
    table_buf: list[str] = []
    in_table = False
    section_has_graph = False
    pending_input: list[str] | None = None
    force_table_main = False
    pending_figure_caption: str | None = None
    skip_next_table = False
    details_toggle_label: str | None = None

    def emit_pending_input_only() -> None:
        nonlocal pending_input
        if pending_input is None:
            return
        out.append(
            '<div class="toggle-row">'
            "<details class=\"input-toggle\">"
            "<summary>View Input</summary>"
            f"{render_input_block(pending_input)}"
            "</details>"
            "</div>"
        )
        pending_input = None

    def flush_table() -> None:
        nonlocal table_buf, in_table, pending_input, force_table_main, skip_next_table
        nonlocal details_toggle_label
        if not table_buf:
            in_table = False
            return
        if skip_next_table:
            table_buf = []
            in_table = False
            skip_next_table = False
            force_table_main = False
            details_toggle_label = None
            return
        table_html = render_table(table_buf)

        if details_toggle_label is not None:
            label = details_toggle_label
            details_toggle_label = None
            out.append(
                "<details class=\"table-toggle\">"
                f"<summary>{H.escape(label)}</summary>"
                f"{table_html}"
                "</details>"
            )
            table_buf = []
            in_table = False
            force_table_main = False
            return

        use_toggle = section_has_graph and not force_table_main

        row_parts: list[str] = []
        if use_toggle:
            row_parts.append(
                "<details class=\"table-toggle\">"
                "<summary>Show table</summary>"
                f"{table_html}"
                "</details>"
            )
        if pending_input is not None:
            row_parts.append(
                "<details class=\"input-toggle\">"
                "<summary>View Input</summary>"
                f"{render_input_block(pending_input)}"
                "</details>"
            )
            pending_input = None

        if not use_toggle:
            out.append(table_html)
        if row_parts:
            out.append('<div class="toggle-row">' + "".join(row_parts) + "</div>")

        table_buf = []
        in_table = False
        force_table_main = False

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Collect Input: block (indented continuation lines)
        if line.startswith("- Input:"):
            first = line.split("Input:", 1)[1].strip()
            buf: list[str] = []
            if first:
                buf.append(first)
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("  ") or nxt.startswith("\t"):
                    buf.append(nxt[2:] if nxt.startswith("  ") else nxt[1:])
                    i += 1
                    continue
                if nxt.strip() == "":
                    # peek: blank line inside input vs end
                    j = i + 1
                    if j < len(lines) and (
                        lines[j].startswith("  ") or lines[j].startswith("\t")
                    ):
                        buf.append("")
                        i += 1
                        continue
                    break
                break
            pending_input = buf
            continue

        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            if not in_table:
                in_table = True
                table_buf = []
            table_buf.append(line)
            i += 1
            continue
        if in_table:
            flush_table()

        if line.startswith("# "):
            # Document H1 is mirrored in the sticky header
            i += 1
            continue
        if line.startswith("## "):
            emit_pending_input_only()
            section_has_graph = False
            title = line[3:]
            aid = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
            out.append(f'<h2 id="{H.escape(aid)}">{H.escape(title)}</h2>')
        elif line.startswith("### "):
            emit_pending_input_only()
            section_has_graph = False
            out.append(f"<h3>{H.escape(line[4:])}</h3>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{H.escape(line[2:])}</blockquote>")
        elif line.strip() == "---":
            emit_pending_input_only()
            out.append("<hr>")
        elif line.startswith("- Graph:"):
            section_has_graph = True
            path = line.split("Graph:", 1)[1].strip()
            href = href_for(path)
            label = H.escape(Path(_clean_path(path)).name)
            if pending_figure_caption:
                cap = inline_fmt(pending_figure_caption)
                pending_figure_caption = None
            else:
                cap = H.escape(_clean_path(path))
            if href:
                out.append(
                    f'<figure class="graph"><img src="{H.escape(href)}" alt="{label}" loading="lazy">'
                    f"<figcaption>{cap}</figcaption></figure>"
                )
            else:
                out.append(f'<p class="missing">Missing graph: {H.escape(path)}</p>')
        elif line.startswith("- Html: skip") or line.startswith("- HTML: skip"):
            skip_next_table = True
        elif line.startswith("- RawHtml:") or line.startswith("- RawHTML:"):
            # Indented continuation lines are raw HTML (for multicol table variants).
            first = line.split(":", 1)[1].strip()
            buf: list[str] = []
            if first:
                buf.append(first)
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("  ") or nxt.startswith("\t"):
                    buf.append(nxt[2:] if nxt.startswith("  ") else nxt[1:])
                    i += 1
                    continue
                break
            out.append("\n".join(buf))
            continue
        elif line.startswith("- DetailsToggle:") or line.startswith("- Detailstoggle:"):
            details_toggle_label = line.split(":", 1)[1].strip() or "Details"
        elif line.startswith("- Finding:"):
            out.append(
                f'<p class="finding"><b>Finding.</b> {inline_fmt(line.split("Finding:", 1)[1].strip())}</p>'
            )
        elif line.startswith("- Detail:"):
            out.append(
                f'<p class="finding"><b>Finding.</b> {inline_fmt(line.split("Detail:", 1)[1].strip())}</p>'
            )
        elif line.startswith(
            ("- Source:", "- Data:", "- Scores:", "- HTML:", "- Hardness", "- Data JSON:", "- Labels:")
        ):
            out.append(f'<p class="path">{H.escape(line[2:])}</p>')
        elif line.startswith("- "):
            out.append(f"<li>{inline_fmt(line[2:])}</li>")
        elif line.strip() == "":
            out.append("")
        elif line.startswith("**Figure ") or line.startswith("**그림 "):
            # Caption is attached to the following Graph (toggle tables stay unnumbered).
            m = re.match(r"\*\*(.+?)\*\*\s*(.*)$", line)
            if m:
                label, rest = m.group(1).rstrip("."), m.group(2).strip()
                pending_figure_caption = f"{label}. {rest}".strip() if rest else f"{label}."
            else:
                pending_figure_caption = line.strip("*")
        elif line.startswith("**Table ") or line.startswith("**표 "):
            # Numbered paper tables stay in the main flow (not under Show table).
            force_table_main = True
            section_has_graph = False
            m = re.match(r"\*\*(.+?)\*\*\s*(.*)$", line)
            if m:
                out.append(
                    f"<h4>{H.escape(m.group(1))} "
                    f'<span class="sub">{H.escape(m.group(2))}</span></h4>'
                )
            else:
                out.append(f"<p>{inline_fmt(line)}</p>")
        elif re.match(r"^\*\*[^*]+\*\*\s*$", line):
            # Standalone bold title at Table-heading level (e.g. Blank Accuracy).
            title = line.strip().strip("*").strip()
            out.append(f"<h4>{H.escape(title)}</h4>")
        else:
            out.append(f"<p>{inline_fmt(line)}</p>")
        i += 1

    if in_table:
        flush_table()
    emit_pending_input_only()

    html_body: list[str] = []
    lis: list[str] = []

    def flush_lis() -> None:
        nonlocal lis
        if lis:
            html_body.append('<ul class="details-list">' + "".join(lis) + "</ul>")
            lis = []

    for x in out:
        if x.startswith("<li>"):
            lis.append(x)
        else:
            flush_lis()
            html_body.append(x)
    flush_lis()
    return "\n".join(html_body)


CSS = """
:root {
  --bg:#ffffff; --ink:#000000; --muted:#333333; --line:#000000;
  --card:#fff; --accent:#1f6feb; --claim:#000000;
}
* { box-sizing:border-box; }
body {
  margin:0; font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  background:var(--bg); color:var(--ink); line-height:1.55;
}
header {
  position:sticky; top:0; z-index:10; background:#ffffffe8;
  border-bottom:1px solid var(--line); padding:14px 28px; backdrop-filter:blur(6px);
}
header .product {
  margin:0; font-size:.78rem; letter-spacing:.04em; text-transform:uppercase;
  color:var(--muted); font-family:ui-sans-serif,system-ui,sans-serif;
}
header h1 { margin:4px 0 0; font-size:1.15rem; font-weight:600; }
header .arc { margin:4px 0 0; color:var(--accent); font-style:italic; font-size:.95rem; }
nav { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
nav a {
  color:#000; text-decoration:none; font-size:.8rem;
  border:1px solid #000; padding:3px 10px; border-radius:0; background:#fff;
}
main { max-width:980px; margin:0 auto; padding:24px 28px 80px; }
h2 {
  margin:2.2rem 0 .6rem; font-size:1.35rem; color:var(--claim);
  border-top:2px solid var(--claim); padding-top:1rem;
}
h3 { margin:1.4rem 0 .5rem; font-size:1.08rem; letter-spacing:.01em; font-weight:600; }
h4 { margin:1.2rem 0 .45rem; font-size:.98rem; }
h4 .sub { font-weight:400; color:var(--muted); font-size:.9rem; }
p { margin:.45rem 0; }
ul.details-list {
  margin:.35rem 0 1rem 1.15rem; color:#222;
  font-family:ui-sans-serif,system-ui,sans-serif; font-size:.9rem;
}
ul.details-list li { margin:.35rem 0; line-height:1.45; }
blockquote {
  margin:1rem 0; padding:.7rem 1rem; border-left:4px solid var(--accent);
  background:#fff; font-size:1.05rem;
}
table {
  width:100%; border-collapse:collapse; background:var(--card);
  font-family:ui-sans-serif,system-ui,sans-serif; font-size:.84rem;
  margin:.6rem 0 0.4rem; box-shadow:0 1px 0 var(--line);
}
th, td { border:1px solid var(--line); padding:6px 8px; vertical-align:middle; }
th.col-text, td.col-text { text-align:left; }
th.col-num, td.col-num { text-align:right; }
td.col-hi { background:#f0f0f0; }
th.col-hi { background:#2a2a2a; }
th { background:#000; color:#fff; font-weight:600; }
figure.graph {
  margin:.5rem 0 1.0rem; padding:10px; background:#fff;
  border:1px solid #000; border-radius:0;
}
figure.graph img { width:100%; max-width:820px; height:auto; display:block; margin:0 auto; }
figcaption {
  margin-top:8px; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:.88rem; color:var(--ink); text-align:center;
}
p.path { font-family:ui-monospace,Menlo,monospace; font-size:.75rem; color:var(--muted); }
p.finding {
  font-family:ui-sans-serif,system-ui,sans-serif; font-size:.92rem;
  color:#111; background:#fff; border:1px solid #000; padding:10px 12px; margin:.4rem 0 .7rem;
}
div.toggle-row {
  display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start;
  margin:.35rem 0 1.1rem;
}
details.table-toggle, details.input-toggle {
  margin:0; font-family:ui-sans-serif,system-ui,sans-serif; font-size:.88rem;
}
details.table-toggle > summary, details.input-toggle > summary {
  cursor:pointer; color:#111; border:1px solid #000; padding:6px 10px; background:#fff;
  list-style:none; width:fit-content;
}
details.table-toggle > summary::-webkit-details-marker,
details.input-toggle > summary::-webkit-details-marker { display:none; }
details.table-toggle[open] > summary,
details.input-toggle[open] > summary { margin-bottom:.45rem; }
div.input-example {
  margin-top:.35rem; padding:12px 14px; border:1px solid #000; background:#fff;
  font-family:ui-monospace,Menlo,monospace; font-size:.78rem; line-height:1.55;
  white-space:normal; max-width:820px;
}
p.missing { color:#a33; }
code {
  font-family:ui-monospace,Menlo,monospace; font-size:.88em;
  background:#fff; border:1px solid #000; padding:0 4px; border-radius:0;
}
hr { border:none; border-top:1px solid var(--line); margin:1.5rem 0; }
"""


def build_html() -> Path:
    body = md_to_html_body(MD.read_text(encoding="utf-8"))
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{H.escape(TITLE)}</title>
<style>{CSS}</style></head><body>
<header>
  <p class="product">{H.escape(PRODUCT)}</p>
  <h1>{H.escape(TITLE)}</h1>
  <p class="arc">{H.escape(ARC)}</p>
  <nav>
    <a href="#llms-have-abstract-knowledge-of-robot-gesture-attributes">Knowledge</a>
    <a href="#where-generation-breaks">Gap</a>
    <a href="#vlms-also-struggle-when-pose-grounding-is-weak">VLM</a>
    <a href="#protocol-notes">Protocol</a>
  </nav>
</header>
<main>
{body}
</main>
</body></html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")
    return OUT_HTML


def build_pptx() -> Path | None:
    if not HAS_PPTX:
        print("PPTX skipped (python-pptx not installed)", file=sys.stderr)
        return None
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def add_title(title: str, subtitle: str = "") -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(0.7), Inches(2.4), Inches(12), Inches(1.2))
        p = box.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
        if subtitle:
            box2 = slide.shapes.add_textbox(Inches(0.7), Inches(3.7), Inches(12), Inches(1.2))
            p2 = box2.text_frame.paragraphs[0]
            p2.text = subtitle
            p2.font.size = Pt(16)
            p2.font.italic = True
            p2.font.color.rgb = RGBColor(0x1F, 0x6F, 0xEB)

    def add_bullets(title: str, bullets: list[str]) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        t = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.7))
        t.text_frame.paragraphs[0].text = title
        t.text_frame.paragraphs[0].font.size = Pt(24)
        t.text_frame.paragraphs[0].font.bold = True
        body = slide.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.5), Inches(5.8))
        tf = body.text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = b
            p.font.size = Pt(16)
            p.space_after = Pt(10)

    def add_image_slide(title: str, img_path: Path) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        t = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.55))
        t.text_frame.paragraphs[0].text = title
        t.text_frame.paragraphs[0].font.size = Pt(20)
        t.text_frame.paragraphs[0].font.bold = True
        if not img_path.is_file():
            m = slide.shapes.add_textbox(Inches(0.5), Inches(3), Inches(12), Inches(1))
            m.text_frame.paragraphs[0].text = f"Missing: {img_path}"
            return
        slide.shapes.add_picture(str(img_path), Inches(0.55), Inches(0.95), width=Inches(12.2))

    plots = REPO / "data/results/evaluation/plots"
    cue = REPO / "data/results/evaluation/embodiment_cue_id"

    add_title(TITLE, PRODUCT)
    add_bullets(
        "Arc",
        [
            "RQ: Can LLMs turn gesture knowledge into recognizable robot motion?",
            "Yes, but approximately — humans recognize human-authored ≫ Gemini-made",
            "Knowledge: blank-config recovers attributes; static ≈ dynamic; hardness flat",
            "Gap: non-humanoid dynamics; VLM judge mirrors morphology pattern",
        ],
    )
    add_image_slide("Fig1 · Cue Identification by Human", cue / "cueid_human_cue_identification.png")
    add_image_slide("Fig2 · Fill in the blank (static/dynamic)", plots / "h4_blank_stretch_32b_static_dynamic.png")
    add_image_slide("Fig3 · Blank number hardness", plots / "h4_hardness_32b_embodiment_heatmap.png")
    add_image_slide("Fig4 · Yes/no + easy/hard", cue / "cueid_human_binary_acc_by_platform_gemini_made.png")
    add_image_slide("Fig5 · Generation vs Baselines", cue / "cueid_human_cue_identification_with_traj_baselines.png")
    add_image_slide("Fig6 · Non-humanoid needs dynamic", plots / "gt_static_dynamic_component_share.png")
    add_image_slide(
        "Fig7 · Non-humanoid Blank Accuracy",
        plots / "h4_blank_acc_morphology_static_dynamic_core_sub_32b.png",
    )
    add_image_slide("Fig8 · Gemini-judge pairwise / yes-no", cue / "cueid_pairwise_acc_by_platform.png")

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print("slides", len(prs.slides))
    return OUT_PPTX


def main() -> None:
    html = build_html()
    pptx = build_pptx()
    print("HTML", html)
    if pptx:
        print("PPTX", pptx)


if __name__ == "__main__":
    main()
