from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal, Optional
import re
import unicodedata
import ftfy
import math


# ──────────────────────────────────────────────────────────────────────────────
# Precompiled regexes (perf)
# ──────────────────────────────────────────────────────────────────────────────
# Page artifacts
PAGE_OF_RE = re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)
STANDALONE_NUM_RE = re.compile(r"^\s*\d+\s*$")

# Code tagging
FENCED_CODE_RE = re.compile(r"```(.*?)```", re.DOTALL)
INDENT_CODE_RE = re.compile(r"^\s{4,}")
CODE_TOKENS_RE = re.compile(r"(;|\{|\}|#include|def\s|class\s|function\s|var\s|let\s)")
# Code-fence line sentinel (for line-by-line scans)
FENCE_LINE_RE = re.compile(r"^\s*```")

# Table tagging (very light heuristics)
TABLE_PIPE_RE = re.compile(r"\S\s*\|\s*\S")  # lines containing pipes with content
TABLE_DASH_RE = re.compile(r"^\s*-{3,}\s*$")  # horizontal-rule-ish rows

# Hyphenation (word characters incl. Latin-1 letters/digits)
WORD_CHARS = r"A-Za-zÀ-ÖØ-öø-ÿ0-9"
HYPHEN_LINEBREAK_RE = re.compile(rf"([{WORD_CHARS}]+)-\n([{WORD_CHARS}]+)")

# Soft-wrap helpers
SENTENCE_END_RE = re.compile(
    r"[.!?…)](?:['\"\u00BB»])?\s*$"
)  # ends with sentence-ish punctuation
BULLET_LINE_RE = re.compile(
    r"^\s*([\-–—•·*]\s+|\d+\.\s+|[a-zA-Z]\)\s+)"
)  # bullets/enumerations
ALL_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s\-_&]+$")

# Stronger type aliases for static checkers (Pylance/mypy)
NormalizationForm = Literal["NFC", "NFKC", "NFD", "NFKD"]

# Bullet and symbols cleanup
ALNUM_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")
BULLET_GLYPHS = "•◦○◯●▪▫■□◻◼⁃·"
RULE_CHARS = "-–—_=*.•·"
SYMBOL_ONLY_RE = re.compile(rf"^[\s{re.escape(BULLET_GLYPHS + RULE_CHARS)}]+$")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PreprocessConfig:
    # Normalization
    use_ftfy: bool = True
    normalize_form: NormalizationForm = "NFKC"  # "NFC" for conservative normalization

    # Headers/Footers
    remove_headers_footers: bool = True
    header_footer_lines_top: int = 2
    header_footer_lines_bottom: int = 2
    header_footer_freq_threshold: float = 0.5  # appears on >= 50% pages

    # Page artifacts
    remove_page_artifacts: bool = True
    page_num_head_window: int = 3  # first N lines
    page_num_tail_window: int = 4  # last N lines

    # Hyphenation
    hyphenation_strategy: str = "smart"  # {"smart","merge","space","keep"}
    hyphenation_keep_smallwords: bool = True
    apply_hyphenation_pdf_only: bool = True

    # Soft wraps
    apply_softwrap_pdf_only: bool = True

    # Tagging
    tag_tables: bool = True
    tag_code: bool = True

    # Bullet / symbol-only cleanup
    clean_symbol_only_lines: bool = (
        True  # merge bullets with following text (lookahead=1) or drop as noise
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_document(
    pages: List[str],
    cfg: PreprocessConfig,
    doc_meta: Optional[Dict[str, str]] = None,
    doc_type: str = "pdf",  # "pdf" | "docx" | "txt" (more later)
) -> Tuple[str, List[str]]:
    """
    Preprocess a multi-page document (already text-extracted).
    Returns:
        full_text: concatenated preprocessed text
        page_texts: list of per-page processed text
    """
    if not pages:
        return "", []

    # 1) Normalize each page early (ftfy + Unicode)
    norm_pages = []
    for i, p in enumerate(pages):
        t = _normalize_text(p, cfg)
        norm_pages.append(t)

    # 2) Optionally detect repeating headers/footers across pages
    headers_to_drop: set[str] = set()
    footers_to_drop: set[str] = set()
    if cfg.remove_headers_footers and len(norm_pages) >= 3:
        headers_to_drop, footers_to_drop = _detect_repeating_headers_footers(
            norm_pages, cfg
        )

    # 3) Per-page cleanup
    out_pages: List[str] = []
    for idx, page in enumerate(norm_pages, start=1):
        t = page

        # Remove repeating headers/footers (exact match policy)
        if cfg.remove_headers_footers and (headers_to_drop or footers_to_drop):
            t = _strip_headers_footers(
                t,
                headers_to_drop,
                footers_to_drop,
                cfg.header_footer_lines_top,
                cfg.header_footer_lines_bottom,
            )

        if cfg.remove_page_artifacts:
            t = _strip_page_artifacts(t, cfg)

        # Hyphenation + soft wraps typically only needed for PDFs
        if _should_apply_hyphenation(doc_type, cfg):
            t = _join_hyphenated_linebreaks(t, cfg)

        if _should_apply_softwrap(doc_type, cfg):
            t = _repair_soft_wraps(t)

        if cfg.tag_tables:
            t = _mark_table_blocks(t)

        if cfg.tag_code:
            t = _mark_code_blocks(t)

        if cfg.clean_symbol_only_lines:
            t = _clean_symbol_only_and_bullets(t)

        t = _final_whitespace_cleanup(t)

        out_pages.append(t)

    # 4) Join with explicit page separators to preserve rough structure
    full_text = "\n\n".join(out_pages)
    return full_text, out_pages


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: normalization
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_text(text: str, cfg: PreprocessConfig) -> str:
    t = text

    # Fix mojibake and oddities
    t2 = ftfy.fix_text(t)
    t = t2

    # Normalize newlines and Unicode
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    try:
        t2 = unicodedata.normalize(cfg.normalize_form, t)
        t = t2
    except Exception:
        # Fallback to NFKC if user passed something odd
        t = unicodedata.normalize("NFKC", t)
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: header/footer detection and removal (exact match)
# ──────────────────────────────────────────────────────────────────────────────
def _first_nonempty(lines: List[str], n: int) -> List[str]:
    out: List[str] = []
    for ln in lines:
        if ln.strip():
            out.append(ln)
        if len(out) == n:
            break
    return out


def _last_nonempty(lines: List[str], n: int) -> List[str]:
    out: List[str] = []
    for ln in reversed(lines):
        if ln.strip():
            out.append(ln)
        if len(out) == n:
            break
    return list(reversed(out))


def _detect_repeating_headers_footers(
    pages: List[str], cfg: PreprocessConfig
) -> Tuple[set[str], set[str]]:
    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}
    total = len(pages)

    for p in pages:
        lines = p.split("\n")
        tops = _first_nonempty(lines, cfg.header_footer_lines_top)
        bots = _last_nonempty(lines, cfg.header_footer_lines_bottom)
        for ln in tops:
            top_counts[ln] = top_counts.get(ln, 0) + 1
        for ln in bots:
            bot_counts[ln] = bot_counts.get(ln, 0) + 1

    cutoff = max(1, math.ceil(total * cfg.header_footer_freq_threshold))
    headers = {ln for ln, c in top_counts.items() if c >= cutoff}
    footers = {ln for ln, c in bot_counts.items() if c >= cutoff}
    return headers, footers


def _strip_headers_footers(
    text: str,
    headers: set[str],
    footers: set[str],
    top_n: int = 2,
    bottom_n: int = 2,
) -> str:
    lines = text.split("\n")

    # collect indices of first/last N NON-EMPTY lines
    top_idx = []
    for i, ln in enumerate(lines):
        if ln.strip():
            top_idx.append(i)
            if len(top_idx) == top_n:
                break
    bot_idx = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            bot_idx.append(i)
            if len(bot_idx) == bottom_n:
                break

    # blank only within those windows if they match detected header/footer strings
    for i in top_idx:
        if lines[i] in headers:
            lines[i] = ""
    for i in bot_idx:
        if lines[i] in footers:
            lines[i] = ""

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: page artifacts
# ──────────────────────────────────────────────────────────────────────────────
def _strip_page_artifacts(text: str, cfg: PreprocessConfig) -> str:
    lines = text.split("\n")
    n = len(lines)
    head_w = min(cfg.page_num_head_window, n)
    tail_w = min(cfg.page_num_tail_window, n)

    for i in range(head_w):
        if PAGE_OF_RE.match(lines[i]) or STANDALONE_NUM_RE.match(lines[i]):
            lines[i] = ""

    for i in range(n - tail_w, n):
        if i < 0 or i >= n:
            continue
        if PAGE_OF_RE.match(lines[i]) or STANDALONE_NUM_RE.match(lines[i]):
            lines[i] = ""

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: hyphenation repair
# ──────────────────────────────────────────────────────────────────────────────
_SMALLWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "nor",
    "of",
    "on",
    "or",
    "out",
    "the",
    "to",
    "via",
    "with",
}


def _join_hyphenated_linebreaks(text: str, cfg: PreprocessConfig) -> str:
    """
    Repairs hyphen + newline splits.
    Strategy:
      - "merge":   left-right       → left + right
      - "space":   left-right       → left + ' ' + right
      - "keep":    leave as-is
      - "smart":   default:
           * if right is a small function word → keep hyphen ("state-of-the-art")
           * if the line already contains another hyphen before the break → space (compound continuation)
           * otherwise merge (true within-word split)
    """
    if cfg.hyphenation_strategy == "keep":
        return text

    # Remove soft hyphen chars; they often appear near line breaks
    t = text.replace("\u00ad", "")

    # Replacement needs access to the original string to inspect context (line-level).
    src = t

    def repl(m: re.Match) -> str:
        left = m.group(1)
        right = m.group(2)

        if cfg.hyphenation_strategy == "merge":
            return f"{left}{right}"
        if cfg.hyphenation_strategy == "space":
            return f"{left} {right}"

        # SMART behavior
        if cfg.hyphenation_keep_smallwords and right.lower() in _SMALLWORDS:
            return f"{left}-{right}"

        # If earlier in the same line we already saw a hyphen, treat as compound continuation → space
        line_start = src.rfind("\n", 0, m.start())
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        line_text = src[line_start : m.start()]
        if "-" in line_text:
            return f"{left} {right}"

        # Otherwise, it's likely a true within-word split → merge
        return f"{left}{right}"

    # Apply once over the whole page
    return HYPHEN_LINEBREAK_RE.sub(repl, t)


def _should_apply_hyphenation(doc_type: str, cfg: PreprocessConfig) -> bool:
    return (
        doc_type.lower() == "pdf" and cfg.apply_hyphenation_pdf_only
    ) or not cfg.apply_hyphenation_pdf_only


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: soft line-wrap repair
# ──────────────────────────────────────────────────────────────────────────────
def _repair_soft_wraps(text: str) -> str:
    """
    Join lines where a hard newline is likely a soft wrap.
    Heuristics (conservative):
      - Do NOT join if the current line looks like a bullet, is ALL CAPS (heading), or ends with ':'.
      - Join only if the current line does NOT end a sentence, and the next line starts lowercase.
    """
    lines = text.split("\n")
    out: List[str] = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if i == len(lines) - 1:
            out.append(ln)
            break

        nxt = lines[i + 1]

        # Guards: bullets/headings/colon-ended
        if (
            BULLET_LINE_RE.match(ln)
            or ALL_CAPS_RE.match(ln.strip())
            or ln.rstrip().endswith(":")
        ):
            out.append(ln)
            i += 1
            continue

        # If current line ends a sentence, keep the break
        if SENTENCE_END_RE.search(ln.rstrip()):
            out.append(ln)
            i += 1
            continue

        # If next line starts with uppercase (consider first alphabetic char), likely a new sentence/paragraph
        def _first_alpha(s: str) -> str:
            for ch in s:
                if ch.isalpha():
                    return ch
            return ""

        alpha = _first_alpha(nxt)
        starts_lower = bool(alpha) and alpha.islower()

        if not starts_lower:
            out.append(ln)
            i += 1
            continue

        # Join ln + nxt with a space
        joined = (ln.rstrip() + " " + nxt.lstrip()).strip()
        out.append(joined)
        i += 2

    return "\n".join(out)


def _should_apply_softwrap(doc_type: str, cfg: PreprocessConfig) -> bool:
    return (
        doc_type.lower() == "pdf" and cfg.apply_softwrap_pdf_only
    ) or not cfg.apply_softwrap_pdf_only


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: table / code tagging (light)
# ──────────────────────────────────────────────────────────────────────────────
def _mark_table_blocks(text: str) -> str:
    """
    Safer table tagging:
      • Skip inside code (``` fences or existing [CODE]…[/CODE]).
      • Require >=2 consecutive 'content' rows with consistent columns.
      • Support '|' or TAB-delimited rows.
      • Markdown dashed separators are allowed *inside* a block but don't count as content rows.
    """
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    in_code = False

    def _pipe_cols(s: str) -> Optional[int]:
        # cheap checks before any split
        first = s.find("|")
        if first == -1:
            return None
        last = s.rfind("|")
        if first == 0 or last == len(s) - 1:
            return None
        # ensure non-empty text on both sides (fast)
        if not s[:first].strip() or not s[last + 1 :].strip():
            return None
        # consistent column count = pipes + 1
        return s.count("|") + 1

    def _tab_cols(s: str) -> Optional[int]:
        return (s.count("\t") + 1) if "\t" in s else None

    while i < len(lines):
        ln = lines[i]

        # Track code regions to avoid tagging inside them
        if ln.startswith("[CODE]") or FENCE_LINE_RE.match(ln):
            in_code = True
            out.append(ln)
            i += 1
            continue
        if ln.startswith("[/CODE]") or (in_code and FENCE_LINE_RE.match(ln)):
            in_code = False
            out.append(ln)
            i += 1
            continue
        if in_code:
            out.append(ln)
            i += 1
            continue

        # Candidate row type?
        pipe_cols = _pipe_cols(ln)
        tab_cols = _tab_cols(ln)

        if pipe_cols is None and tab_cols is None:
            out.append(ln)
            i += 1
            continue

        # Determine delimiter type & expected column count
        delim = "pipe" if pipe_cols is not None else "tab"
        expect_cols = pipe_cols if pipe_cols is not None else tab_cols

        # Grow block with same-type rows; require consistent columns for 'content' rows
        j = i
        content_rows = 0
        block: List[str] = []
        while j < len(lines):
            lnj = lines[j]

            # Stop at code starts to avoid spanning into code
            if (
                lnj.startswith("[CODE]")
                or lnj.startswith("[/CODE]")
                or FENCE_LINE_RE.match(lnj)
            ):
                break

            if delim == "pipe":
                cols = _pipe_cols(lnj)
                if cols is None:
                    # allow dashed separators but don't count them as content
                    if not TABLE_DASH_RE.match(lnj):
                        break
                else:
                    if cols != expect_cols:
                        break
                    content_rows += 1
                block.append(lnj)
                j += 1
                continue
            else:  # tab-delimited
                cols = _tab_cols(lnj)
                if cols is None or cols != expect_cols:
                    break
                content_rows += 1
                block.append(lnj)
                j += 1

        # Only tag if we are confident: ≥2 content rows
        if content_rows >= 2:
            out.append("[TABLE]\n" + "\n".join(block) + "\n[/TABLE]")
            i = j
        else:
            # Not confident → leave original line untouched
            out.append(ln)
            i += 1

    return "\n".join(out)


def _mark_code_blocks(text: str) -> str:
    """
    1) Convert fenced ``` blocks to [CODE]…[/CODE].
    2) Then scan remaining lines for code-ish regions (indented/token-heavy),
       without re-tagging inside existing [CODE] blocks.
    """
    t = FENCED_CODE_RE.sub(r"[CODE]\n\1\n[/CODE]", text)

    lines = t.split("\n")
    out: List[str] = []
    in_code = False
    in_table = False
    i = 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("[CODE]"):
            in_code = True
            out.append(ln)
            i += 1
            continue
        if ln.startswith("[/CODE]"):
            in_code = False
            out.append(ln)
            i += 1
            continue
        if ln.startswith("[TABLE]"):
            in_table = True
            out.append(ln)
            i += 1
            continue
        if ln.startswith("[/TABLE]"):
            in_table = False
            out.append(ln)
            i += 1
            continue

        if in_code or in_table:
            out.append(ln)
            i += 1
            continue

        is_codeish = bool(INDENT_CODE_RE.search(ln)) or bool(CODE_TOKENS_RE.search(ln))
        if is_codeish:
            j = i
            while j < len(lines):
                lnj = lines[j]
                if (
                    lnj.startswith("[CODE]")
                    or lnj.startswith("[/CODE]")
                    or lnj.startswith("[TABLE]")
                    or lnj.startswith("[/TABLE]")
                ):
                    break
                if not (INDENT_CODE_RE.search(lnj) or CODE_TOKENS_RE.search(lnj)):
                    break
                j += 1
            block = lines[i:j]
            out.append("[CODE] " + "\n".join(block) + "\n[/CODE]")
            i = j
            continue

        out.append(ln)
        i += 1

    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: final whitespace cleanup
# ──────────────────────────────────────────────────────────────────────────────


def _clean_symbol_only_and_bullets(text: str) -> str:
    """
    Fix PDF artifacts:
      - If a line is symbol-only and contains a bullet, and the next non-blank line exists and has text,
        merge: "<bullet> <next line>".
      - If a line is symbol-only without a following text line, drop it (or turn into a blank to preserve spacing).
      - Skip entirely inside [CODE]/[TABLE] blocks.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    in_table = False
    while i < len(lines):
        ln = lines[i]

        # Track explicit blocks to avoid touching their contents
        if ln.startswith("[CODE]"):
            in_code = True
            out.append(ln)
            i += 1
            continue
        if ln.startswith("[/CODE]"):
            in_code = False
            out.append(ln)
            i += 1
            continue
        if ln.startswith("[TABLE]"):
            in_table = True
            out.append(ln)
            i += 1
            continue
        if ln.startswith("[/TABLE]"):
            in_table = False
            out.append(ln)
            i += 1
            continue
        if in_code or in_table:
            out.append(ln)
            i += 1
            continue

        # Only consider symbol-only lines outside code/table
        if SYMBOL_ONLY_RE.match(ln):
            # If this line contains any bullet glyph, try to merge with the *next* non-blank line (lookahead=1)
            if any(ch in BULLET_GLYPHS for ch in ln):
                j = i + 1
                if j < len(lines):
                    nxt = lines[j]
                    if nxt.strip() and ALNUM_RE.search(nxt):
                        # preserve the first bullet glyph from the original line
                        bullet = next((ch for ch in ln if ch in BULLET_GLYPHS), "•")
                        out.append(f"{bullet} {nxt.strip()}")
                        i = j + 1
                        continue
                # No merge target → drop bullet-only line
                i += 1
                continue
            else:
                # Rule/separator line (e.g., ---- or =====). Replace with a single blank line at most.
                if not out or out[-1] != "":
                    out.append("")
                i += 1
                continue

        # Default: passthrough
        out.append(ln)
        i += 1
    return "\n".join(out)


def _final_whitespace_cleanup(text: str) -> str:
    # Collapse 3+ blank lines → 2, and trim trailing spaces
    t = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip("\n ")  # keep single leading/trailing newline out


# ──────────────────────────────────────────────────────────────────────────────
# Tiny test harness (optional) — call from your unit tests
# ──────────────────────────────────────────────────────────────────────────────
def _run_smoke_tests() -> None:  # pragma: no cover
    cfg = PreprocessConfig()

    # Hyphenation: within-word vs compounds and smallwords
    page = "intro-\nduction\nstate-\nof-the-art\ncost-benefit-\nanalysis"
    txt, _ = preprocess_document([page], cfg, doc_type="pdf")
    assert "introduction" in txt
    assert "state-of-the-art" in txt
    assert "cost-benefit analysis" in txt

    # Page artifacts
    page2 = "My Report\nACME Inc.\nPage 2 of 10\n\nFurther details."
    txt2, _ = preprocess_document([page2], cfg, doc_type="pdf")
    assert "Page 2 of 10" not in txt2

    # Code tagging: fenced + indented
    page3 = "text\n```\ndef hello():\n    print('x')\n```\n    def foo():\n        pass\nend"
    txt3, _ = preprocess_document([page3], cfg, doc_type="pdf")
    assert "[CODE]" in txt3 and "[/CODE]" in txt3

    # Soft wrap: join lowercase-starting continuation
    page4 = "This is a line\nthat continues on next.\nBut This is new."
    txt4, _ = preprocess_document([page4], cfg, doc_type="pdf")
    assert "line that continues" in txt4

    # Headers/footers: exact-match on >= 50% pages
    p1 = "HEADER\nBody A\n1"
    p2 = "HEADER\nBody B\n2"
    p3 = "HEADER\nBody C\n3"
    txt5, pages5 = preprocess_document([p1, p2, p3], cfg, doc_type="pdf")
    assert all(not pg.split("\n", 1)[0].strip() for pg in pages5)  # header removed
