import os
import re
from typing import Dict, Optional


_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_NAME_LINE_RE = re.compile(r"^[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,3}$")

_PATH_CV_RE = re.compile(r"\b(cv|resume|curriculum[_\s-]*vitae)\b", re.IGNORECASE)
_TEXT_CV_RE = re.compile(
    r"\b(curriculum vitae|professional summary|work experience|employment history)\b",
    re.IGNORECASE,
)
_PATH_COVER_RE = re.compile(r"\b(cover[_\s-]*letter|motivation[_\s-]*letter)\b", re.IGNORECASE)
_PATH_REFERENCE_RE = re.compile(
    r"\b(reference[_\s-]*letter|recommendation[_\s-]*letter)\b",
    re.IGNORECASE,
)

_NAME_STOPWORDS = {
    "cv",
    "resume",
    "curriculum",
    "vitae",
    "cover",
    "letter",
    "motivation",
    "reference",
    "recommendation",
    "profile",
    "final",
    "draft",
    "v1",
    "v2",
    "copy",
}

_NAME_LINE_EXCLUDE = {
    "curriculum vitae",
    "work experience",
    "professional summary",
    "employment history",
    "education",
    "skills",
    "profile",
    "summary",
}


def _infer_doc_type(path: str, filetype: str, full_text: str) -> Optional[str]:
    path_text = f"{path} {filetype}"
    if _PATH_COVER_RE.search(path_text):
        return "cover_letter"
    if _PATH_REFERENCE_RE.search(path_text):
        return "reference_letter"
    if _PATH_CV_RE.search(path_text):
        return "cv"
    if _TEXT_CV_RE.search(full_text[:3000]):
        return "cv"
    return None


def _extract_person_name_from_filename(path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    tokens = [tok for tok in _TOKEN_SPLIT_RE.split(stem) if tok]
    name_parts: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in _NAME_STOPWORDS or token.isdigit():
            if len(name_parts) >= 2:
                break
            continue
        if not token[0].isalpha():
            continue
        cleaned = re.sub(r"[^A-Za-z'`-]", "", token)
        if len(cleaned) < 2:
            continue
        name_parts.append(cleaned.capitalize())
        if len(name_parts) >= 3:
            break
    if len(name_parts) >= 2:
        return " ".join(name_parts)
    return None


def _extract_person_name_from_text(full_text: str) -> Optional[str]:
    for raw_line in full_text.splitlines()[:12]:
        line = raw_line.strip()
        if not line or len(line) > 60:
            continue
        if line.lower() in _NAME_LINE_EXCLUDE:
            continue
        if _NAME_LINE_RE.match(line):
            return line
    return None


def _authority_for_doc_type(doc_type: Optional[str]) -> Optional[float]:
    if doc_type == "cv":
        return 1.0
    if doc_type == "cover_letter":
        return 0.85
    if doc_type == "reference_letter":
        return 0.8
    return None


def classify_document(path: str, filetype: str, full_text: str) -> Dict[str, Optional[str | float]]:
    doc_type = _infer_doc_type(path, filetype, full_text)
    person_name: Optional[str] = None
    if doc_type is not None:
        person_name = _extract_person_name_from_filename(path) or _extract_person_name_from_text(full_text)
    authority_rank = _authority_for_doc_type(doc_type)
    return {
        "doc_type": doc_type,
        "person_name": person_name,
        "authority_rank": authority_rank,
    }


__all__ = ["classify_document"]
