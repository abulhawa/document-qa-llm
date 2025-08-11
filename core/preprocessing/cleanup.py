from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import PreprocessConfig

# Hyphenation
WORD_CHARS = r"A-Za-zÀ-ÖØ-öø-ÿ0-9"
HYPHEN_LINEBREAK_RE = re.compile(rf"([{WORD_CHARS}]+)-\n([{WORD_CHARS}]+)")
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

# Soft-wrap helpers
SENTENCE_END_RE = re.compile(r"[.!?…)](?:['\"\u00BB»])?\s*$")
BULLET_LINE_RE = re.compile(r"^\s*([\-–—•·*]\s+|\d+\.\s+|[a-zA-Z]\)\s+)")
ALL_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s\-_&]+$")

# Bullet and symbols cleanup
ALNUM_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")
BULLET_GLYPHS = "•◦○◯●▪▫■□◻◼⁃·"
RULE_CHARS = "-–—_=*.•·"
SYMBOL_ONLY_RE = re.compile(rf"^[\s{re.escape(BULLET_GLYPHS + RULE_CHARS)}]+$")


def _join_hyphenated_linebreaks(text: str, cfg: "PreprocessConfig") -> str:
    if cfg.hyphenation_strategy == "keep":
        return text
    t = text.replace("\u00ad", "")
    src = t

    def repl(m: re.Match) -> str:
        left = m.group(1)
        right = m.group(2)
        if cfg.hyphenation_strategy == "merge":
            return f"{left}{right}"
        if cfg.hyphenation_strategy == "space":
            return f"{left} {right}"
        if cfg.hyphenation_keep_smallwords and right.lower() in _SMALLWORDS:
            return f"{left}-{right}"
        line_start = src.rfind("\n", 0, m.start())
        line_start = 0 if line_start == -1 else line_start + 1
        line_text = src[line_start : m.start()]
        if "-" in line_text:
            return f"{left} {right}"
        return f"{left}{right}"

    return HYPHEN_LINEBREAK_RE.sub(repl, t)


def _should_apply_hyphenation(doc_type: str, cfg: "PreprocessConfig") -> bool:
    return (
        doc_type.lower() == "pdf" and cfg.apply_hyphenation_pdf_only
    ) or not cfg.apply_hyphenation_pdf_only


def _repair_soft_wraps(text: str) -> str:
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if i == len(lines) - 1:
            out.append(ln)
            break
        nxt = lines[i + 1]
        if (
            BULLET_LINE_RE.match(ln)
            or ALL_CAPS_RE.match(ln.strip())
            or ln.rstrip().endswith(":")
        ):
            out.append(ln)
            i += 1
            continue
        if SENTENCE_END_RE.search(ln.rstrip()):
            out.append(ln)
            i += 1
            continue

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
        joined = (ln.rstrip() + " " + nxt.lstrip()).strip()
        out.append(joined)
        i += 2
    return "\n".join(out)


def _should_apply_softwrap(doc_type: str, cfg: "PreprocessConfig") -> bool:
    return (
        doc_type.lower() == "pdf" and cfg.apply_softwrap_pdf_only
    ) or not cfg.apply_softwrap_pdf_only


def _clean_symbol_only_and_bullets(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    in_table = False
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
        if SYMBOL_ONLY_RE.match(ln):
            if any(ch in BULLET_GLYPHS for ch in ln):
                j = i + 1
                if j < len(lines):
                    nxt = lines[j]
                    if nxt.strip() and ALNUM_RE.search(nxt):
                        bullet = next((ch for ch in ln if ch in BULLET_GLYPHS), "•")
                        out.append(f"{bullet} {nxt.strip()}")
                        i = j + 1
                        continue
                i += 1
                continue
            else:
                if not out or out[-1] != "":
                    out.append("")
                i += 1
                continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def _final_whitespace_cleanup(text: str) -> str:
    t = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip("\n ")
