"""
ingestion/doc_classifier.py
===========================
Classifies a document into a type (cv, research_paper, invoice, etc.) using
rule-based pattern matching on the file path and the first 4000 characters of text.

This runs early in the ingestion pipeline (orchestrator.py calls it right after
full text is extracted) and its output — doc_type, doc_type_confidence — is
merged into every chunk and the full-text document stored in OpenSearch.

The doc_type is also passed to financial_extractor.py as a prior, so getting
this right matters: a correct "research_paper" label prevents the financial
extractor from misclassifying an engineering paper as a tax document.

Classification logic
--------------------
- Path rules run first and carry higher confidence (0.94-0.98).
  Rationale: filenames like "invoice_2023.pdf" or "cv_ali.pdf" are highly reliable.
- Text rules run second and carry lower confidence (0.76-0.88).
  Rationale: keywords in body text are less specific than filename keywords.
- First match wins — order of checks matters. More specific types (cover_letter,
  reference_letter) are checked before generic ones (cv, research_paper).
- Falls back to doc_type="other" with confidence 0.25 if nothing matches.

Output dict keys
----------------
- doc_type: str            e.g. "cv", "research_paper", "invoice", "other"
- doc_type_confidence: float  0.25-0.98
- doc_type_source: str     "rule" or "fallback"
- person_name: Optional[str]  extracted for CVs, cover letters, reference letters
- authority_rank: Optional[float]  relevance weight for person-centric doc types
"""

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
_PATH_RESEARCH_PAPER_RE = re.compile(
    r"\b(research[_\s-]*paper|journal[_\s-]*paper|conference[_\s-]*paper|paper)\b",
    re.IGNORECASE,
)
_TEXT_RESEARCH_PAPER_RE = re.compile(
    r"\b(abstract|introduction|methodology|results|discussion|references|doi)\b",
    re.IGNORECASE,
)
_PATH_TECHNICAL_REPORT_RE = re.compile(
    r"\b(technical[_\s-]*report|project[_\s-]*report|feasibility[_\s-]*study|report)\b",
    re.IGNORECASE,
)
_TEXT_TECHNICAL_REPORT_RE = re.compile(
    r"\b(project report|technical report|system design|implementation|case study)\b",
    re.IGNORECASE,
)
_PATH_COURSE_MATERIAL_RE = re.compile(
    r"\b(course|lecture|syllabus|slides|assignment|tutorial|module)\b",
    re.IGNORECASE,
)
_TEXT_COURSE_MATERIAL_RE = re.compile(
    r"\b(course code|lecture notes|syllabus|assignment|tutorial|semester)\b",
    re.IGNORECASE,
)
_PATH_CONTRACT_RE = re.compile(
    r"\b(contract|agreement|lease|terms[_\s-]*and[_\s-]*conditions)\b",
    re.IGNORECASE,
)
_TEXT_CONTRACT_RE = re.compile(
    r"\b(this agreement|party of the first part|terms and conditions|effective date)\b",
    re.IGNORECASE,
)
_PATH_POLICY_RE = re.compile(r"\b(policy|guideline|compliance|handbook)\b", re.IGNORECASE)
_TEXT_POLICY_RE = re.compile(
    r"\b(policy statement|compliance|scope|purpose|applies to)\b",
    re.IGNORECASE,
)
_PATH_INVOICE_RE = re.compile(r"\b(invoice|receipt|rechnung|bill)\b", re.IGNORECASE)
_TEXT_INVOICE_RE = re.compile(
    r"\b(invoice number|bill to|total due|vat|amount due)\b",
    re.IGNORECASE,
)
_PATH_PAYROLL_RE = re.compile(
    r"\b(payroll|payslip|salary|wage|lohnabrechnung|gehaltsabrechnung)\b",
    re.IGNORECASE,
)
_TEXT_PAYROLL_RE = re.compile(
    r"\b(gross pay|net pay|salary statement|pay period|tax withheld)\b",
    re.IGNORECASE,
)
_PATH_INSURANCE_RE = re.compile(
    r"\b(insurance|insurer|premium|claim|versicherung)\b",
    re.IGNORECASE,
)
_TEXT_INSURANCE_RE = re.compile(
    r"\b(policyholder|premium|coverage|claim number|insurance)\b",
    re.IGNORECASE,
)
_PATH_GOV_FORM_RE = re.compile(
    r"\b(form|formular|jobcenter|government|behorde|beh\u00f6rde|amt|application)\b",
    re.IGNORECASE,
)
_TEXT_GOV_FORM_RE = re.compile(
    r"\b(application form|jobcenter|antrag|kundennummer|case number|government office)\b",
    re.IGNORECASE,
)
_PATH_ACADEMIC_RECORD_RE = re.compile(
    r"\b(transcript|academic[_\s-]*record|certificate|diploma|grade[_\s-]*report)\b",
    re.IGNORECASE,
)
_TEXT_ACADEMIC_RECORD_RE = re.compile(
    r"\b(transcript|gpa|academic record|certificate|matriculation)\b",
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


def _infer_doc_type(path: str, filetype: str, full_text: str) -> tuple[Optional[str], Optional[float], str]:
    """
    Core classification logic. Returns (doc_type, confidence, source).
    Checks path+filename patterns first (higher confidence), then text patterns.
    Order of checks is significant — more specific document types are matched first.
    """
    path_text = re.sub(r"[_/\\.\-]+", " ", f"{path} {filetype}")
    text_sample = full_text[:4000]

    if _PATH_COVER_RE.search(path_text):
        return "cover_letter", 0.98, "rule"
    if _PATH_REFERENCE_RE.search(path_text):
        return "reference_letter", 0.98, "rule"
    if _PATH_CV_RE.search(path_text):
        return "cv", 0.97, "rule"
    if _TEXT_CV_RE.search(text_sample):
        return "cv", 0.88, "rule"

    if _PATH_RESEARCH_PAPER_RE.search(path_text):
        return "research_paper", 0.95, "rule"
    if _TEXT_RESEARCH_PAPER_RE.search(text_sample):
        return "research_paper", 0.78, "rule"

    if _PATH_TECHNICAL_REPORT_RE.search(path_text):
        return "technical_report", 0.94, "rule"
    if _TEXT_TECHNICAL_REPORT_RE.search(text_sample):
        return "technical_report", 0.77, "rule"

    if _PATH_COURSE_MATERIAL_RE.search(path_text):
        return "course_material", 0.94, "rule"
    if _TEXT_COURSE_MATERIAL_RE.search(text_sample):
        return "course_material", 0.76, "rule"

    if _PATH_CONTRACT_RE.search(path_text):
        return "contract", 0.96, "rule"
    if _TEXT_CONTRACT_RE.search(text_sample):
        return "contract", 0.80, "rule"

    if _PATH_POLICY_RE.search(path_text):
        return "policy", 0.95, "rule"
    if _TEXT_POLICY_RE.search(text_sample):
        return "policy", 0.78, "rule"

    if _PATH_INVOICE_RE.search(path_text):
        return "invoice", 0.96, "rule"
    if _TEXT_INVOICE_RE.search(text_sample):
        return "invoice", 0.82, "rule"

    if _PATH_PAYROLL_RE.search(path_text):
        return "payroll", 0.96, "rule"
    if _TEXT_PAYROLL_RE.search(text_sample):
        return "payroll", 0.82, "rule"

    if _PATH_INSURANCE_RE.search(path_text):
        return "insurance_letter", 0.95, "rule"
    if _TEXT_INSURANCE_RE.search(text_sample):
        return "insurance_letter", 0.80, "rule"

    if _PATH_GOV_FORM_RE.search(path_text):
        return "government_form", 0.95, "rule"
    if _TEXT_GOV_FORM_RE.search(text_sample):
        return "government_form", 0.80, "rule"

    if _PATH_ACADEMIC_RECORD_RE.search(path_text):
        return "academic_record", 0.96, "rule"
    if _TEXT_ACADEMIC_RECORD_RE.search(text_sample):
        return "academic_record", 0.82, "rule"

    return "other", 0.25, "fallback"


def _extract_person_name_from_filename(path: str) -> Optional[str]:
    """
    Attempts to extract a person's name from the filename stem.
    Looks for 2-3 consecutive capitalised tokens that are not in the stopword list.
    Example: "CV_Ali_Abul-Hawa_2024.pdf" → "Ali Abul-Hawa"
    """
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
    """
    Scans the first 12 lines of text for a line that looks like a person's name:
    2-4 capitalised words, not in the exclusion list, not longer than 60 chars.
    This is used as a fallback when the filename doesn't yield a name.
    """
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
    """
    Returns a relevance weight for person-centric document types.
    Used downstream to rank results when searching across multiple doc types.
    CVs carry the highest authority (1.0), cover letters slightly less (0.85).
    Returns None for non-person-centric types.
    """
    if doc_type == "cv":
        return 1.0
    if doc_type == "cover_letter":
        return 0.85
    if doc_type == "reference_letter":
        return 0.8
    return None


def classify_document(path: str, filetype: str, full_text: str) -> Dict[str, Optional[str | float]]:
    """
    Public entry point. Classifies a document and returns a metadata dict.
    Called by orchestrator.ingest_one() after full text is extracted.
    The returned dict is merged into both the full-text document and every chunk.
    """
    doc_type, doc_type_confidence, doc_type_source = _infer_doc_type(path, filetype, full_text)
    person_name: Optional[str] = None
    if doc_type in {"cv", "cover_letter", "reference_letter"}:
        person_name = _extract_person_name_from_filename(path) or _extract_person_name_from_text(full_text)
    authority_rank = _authority_for_doc_type(doc_type)
    return {
        "doc_type": doc_type,
        "doc_type_confidence": doc_type_confidence,
        "doc_type_source": doc_type_source,
        "person_name": person_name,
        "authority_rank": authority_rank,
    }


__all__ = ["classify_document"]
