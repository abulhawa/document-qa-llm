from __future__ import annotations

from typing import TYPE_CHECKING
import ftfy
import unicodedata

if TYPE_CHECKING:
    from . import PreprocessConfig


def _normalize_text(text: str, cfg: "PreprocessConfig") -> str:
    t = ftfy.fix_text(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    try:
        t = unicodedata.normalize(cfg.normalize_form, t)
    except Exception:
        t = unicodedata.normalize("NFKC", t)
    return t
