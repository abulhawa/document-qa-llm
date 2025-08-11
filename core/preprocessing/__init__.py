from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import math
import re

from .normalization import _normalize_text
from .tagging import _mark_table_blocks, _mark_code_blocks
from .cleanup import (
    _join_hyphenated_linebreaks,
    _should_apply_hyphenation,
    _repair_soft_wraps,
    _should_apply_softwrap,
    _clean_symbol_only_and_bullets,
    _final_whitespace_cleanup,
)

# Page artifact regexes
PAGE_OF_RE = re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)
STANDALONE_NUM_RE = re.compile(r"^\s*\d+\s*$")

NormalizationForm = Literal["NFC", "NFKC", "NFD", "NFKD"]


@dataclass(slots=True)
class PreprocessConfig:
    # Normalization
    use_ftfy: bool = True
    normalize_form: NormalizationForm = "NFKC"

    # Headers/Footers
    remove_headers_footers: bool = True
    header_footer_lines_top: int = 2
    header_footer_lines_bottom: int = 2
    header_footer_freq_threshold: float = 0.5

    # Page artifacts
    remove_page_artifacts: bool = True
    page_num_head_window: int = 3
    page_num_tail_window: int = 4

    # Hyphenation
    hyphenation_strategy: str = "smart"
    hyphenation_keep_smallwords: bool = True
    apply_hyphenation_pdf_only: bool = True

    # Soft wraps
    apply_softwrap_pdf_only: bool = True

    # Tagging
    tag_tables: bool = True
    tag_code: bool = True

    # Bullet / symbol-only cleanup
    clean_symbol_only_lines: bool = True


def preprocess_document(
    pages: List[str],
    cfg: PreprocessConfig,
    doc_meta: Optional[Dict[str, str]] = None,
    doc_type: str = "pdf",
) -> Tuple[str, List[str]]:
    """Preprocess a multi-page document."""
    if not pages:
        return "", []

    norm_pages = [_normalize_text(p, cfg) for p in pages]

    headers_to_drop: set[str] = set()
    footers_to_drop: set[str] = set()
    if cfg.remove_headers_footers and len(norm_pages) >= 3:
        headers_to_drop, footers_to_drop = _detect_repeating_headers_footers(
            norm_pages, cfg
        )

    out_pages: List[str] = []
    for page in norm_pages:
        t = page
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

    full_text = "\n\n".join(out_pages)
    return full_text, out_pages


# ----------------------------------------------------------------------------
# Header/footer detection and removal
# ----------------------------------------------------------------------------

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

    for i in top_idx:
        if lines[i] in headers:
            lines[i] = ""
    for i in bot_idx:
        if lines[i] in footers:
            lines[i] = ""

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Page artifacts
# ----------------------------------------------------------------------------

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


__all__ = ["PreprocessConfig", "preprocess_document"]
