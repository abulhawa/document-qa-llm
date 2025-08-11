from __future__ import annotations

import re
from typing import List, Optional

# Code tagging
FENCED_CODE_RE = re.compile(r"```(.*?)```", re.DOTALL)
INDENT_CODE_RE = re.compile(r"^\s{4,}")
CODE_TOKENS_RE = re.compile(r"(;|\{|\}|#include|def\s|class\s|function\s|var\s|let\s)")
# Code-fence line sentinel (for line-by-line scans)
FENCE_LINE_RE = re.compile(r"^\s*```")

# Table tagging
TABLE_PIPE_RE = re.compile(r"\S\s*\|\s*\S")
TABLE_DASH_RE = re.compile(r"^\s*-{3,}\s*$")


def _mark_table_blocks(text: str) -> str:
    """Lightweight table tagging that skips code blocks."""
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    in_code = False

    def _pipe_cols(s: str) -> Optional[int]:
        first = s.find("|")
        if first == -1:
            return None
        last = s.rfind("|")
        if first == 0 or last == len(s) - 1:
            return None
        if not s[:first].strip() or not s[last + 1 :].strip():
            return None
        return s.count("|") + 1

    def _tab_cols(s: str) -> Optional[int]:
        return (s.count("\t") + 1) if "\t" in s else None

    while i < len(lines):
        ln = lines[i]

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

        pipe_cols = _pipe_cols(ln)
        tab_cols = _tab_cols(ln)
        if pipe_cols is None and tab_cols is None:
            out.append(ln)
            i += 1
            continue

        delim = "pipe" if pipe_cols is not None else "tab"
        expect_cols = pipe_cols if pipe_cols is not None else tab_cols

        j = i
        content_rows = 0
        block: List[str] = []
        while j < len(lines):
            lnj = lines[j]
            if (
                lnj.startswith("[CODE]")
                or lnj.startswith("[/CODE]")
                or FENCE_LINE_RE.match(lnj)
            ):
                break

            if delim == "pipe":
                cols = _pipe_cols(lnj)
                if cols is None:
                    if not TABLE_DASH_RE.match(lnj):
                        break
                else:
                    if cols != expect_cols:
                        break
                    content_rows += 1
                block.append(lnj)
                j += 1
                continue
            else:
                cols = _tab_cols(lnj)
                if cols is None or cols != expect_cols:
                    break
                content_rows += 1
                block.append(lnj)
                j += 1

        if content_rows >= 2:
            out.append("[TABLE]\n" + "\n".join(block) + "\n[/TABLE]")
            i = j
        else:
            out.append(ln)
            i += 1

    return "\n".join(out)


def _mark_code_blocks(text: str) -> str:
    """Tag code blocks using fences or heuristics."""
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
