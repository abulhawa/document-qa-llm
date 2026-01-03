from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from config import logger
from core.llm import ask_llm, check_llm_status
from core.embeddings import embed_texts
from core.file_loader import load_documents
from utils.file_utils import get_file_size, normalize_path


_TOPIC_PATTERN = re.compile(r"^[1-5]\.\s*")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DEFAULT_EXCLUDE_DIRS = {".tmp.drivedownload", ".tmp.driveupload"}
_DEFAULT_EXCLUDE_PREFIXES = (".",)


@dataclass(frozen=True)
class TargetFolder:
    label: str
    path: str
    keywords: Dict[str, float]


@dataclass
class SortPlanItem:
    path: str
    target_label: str
    target_path: str
    confidence: float
    meta_similarity: float
    content_similarity: float
    keyword_score: float
    reason: str
    second_confidence: Optional[float] = None
    top2_margin: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "target_label": self.target_label,
            "target_path": self.target_path,
            "confidence": round(self.confidence, 4),
            "second_confidence": round(self.second_confidence, 4)
            if self.second_confidence is not None
            else None,
            "top2_margin": round(self.top2_margin, 4) if self.top2_margin is not None else None,
            "meta_similarity": round(self.meta_similarity, 4),
            "content_similarity": round(self.content_similarity, 4),
            "keyword_score": round(self.keyword_score, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SortOptions:
    root: str
    include_content: bool = True
    max_parent_levels: int = 4
    max_content_chars: int = 6000
    max_content_mb: int = 25
    weight_meta: float = 0.55
    weight_content: float = 0.3
    weight_keyword: float = 0.15
    alias_map_text: str = ""
    max_files: Optional[int] = None
    use_llm_fallback: bool = False
    llm_confidence_floor: float = 0.65
    llm_max_items: int = 200


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _normalize_label(topic: str, subtopic: str) -> str:
    topic_name = topic.strip()
    return f"{topic_name}/{subtopic.strip()}"


def _extract_keywords(text: str) -> Dict[str, float]:
    return {tok: 1.0 for tok in _tokenize(text)}


def _merge_keywords(base: Dict[str, float], extra: Dict[str, float]) -> Dict[str, float]:
    merged = dict(base)
    for key, weight in extra.items():
        merged[key] = max(weight, merged.get(key, 0.0))
    return merged


def _parse_alias_map(text: str) -> Dict[str, Dict[str, float]]:
    """Parse alias map lines: target|keyword:weight,keyword."""
    mapping: Dict[str, Dict[str, float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        target, raw_keywords = [part.strip() for part in line.split("|", 1)]
        if not target:
            continue
        keywords: Dict[str, float] = {}
        for raw_kw in raw_keywords.split(","):
            item = raw_kw.strip().lower()
            if not item:
                continue
            if ":" in item:
                kw, weight = [p.strip() for p in item.split(":", 1)]
                try:
                    keywords[kw] = float(weight)
                except ValueError:
                    keywords[kw] = 1.0
            else:
                keywords[item] = 1.0
        if keywords:
            mapping[target.lower()] = keywords
    return mapping


def _list_topic_targets(root: str, alias_map_text: str) -> List[TargetFolder]:
    root_path = Path(root)
    alias_map = _parse_alias_map(alias_map_text)
    targets: List[TargetFolder] = []
    for topic in root_path.iterdir():
        if not topic.is_dir():
            continue
        if not _TOPIC_PATTERN.match(topic.name):
            continue
        subdirs = [d for d in topic.iterdir() if d.is_dir()]
        if not subdirs:
            label = topic.name
            keywords = _extract_keywords(label)
            extra = alias_map.get(label.lower(), {})
            targets.append(
                TargetFolder(label=label, path=str(topic), keywords=_merge_keywords(keywords, extra))
            )
            continue
        for sub in subdirs:
            label = _normalize_label(topic.name, sub.name)
            keywords = _extract_keywords(f"{topic.name} {sub.name}")
            extra = alias_map.get(label.lower(), {})
            if not extra:
                extra = alias_map.get(sub.name.lower(), {})
            targets.append(
                TargetFolder(label=label, path=str(sub), keywords=_merge_keywords(keywords, extra))
            )
    return targets


def _should_skip_dir(name: str) -> bool:
    if name in _DEFAULT_EXCLUDE_DIRS:
        return True
    return any(name.startswith(prefix) for prefix in _DEFAULT_EXCLUDE_PREFIXES)


def _scan_files(root: str, exclude_roots: Iterable[str], max_files: Optional[int]) -> List[str]:
    root = os.path.abspath(root)
    exclude_norm = [os.path.abspath(p) for p in exclude_roots]
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        if any(os.path.commonpath([dirpath, ex]) == ex for ex in exclude_norm):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for name in filenames:
            if name.startswith("~$"):
                continue
            files.append(os.path.join(dirpath, name))
            if max_files is not None and len(files) >= max_files:
                return files
    return files


def _build_meta_text(path: str, root: str, max_parent_levels: int) -> Tuple[str, List[str]]:
    rel = os.path.relpath(path, root)
    parts = Path(rel).parts
    filename = parts[-1]
    parents = list(parts[:-1])[-max_parent_levels:] if max_parent_levels > 0 else []
    meta_text = " ".join([filename] + parents)
    tokens = _tokenize(meta_text)
    return meta_text, tokens


def _load_content_text(path: str, max_chars: int, max_mb: int) -> str:
    try:
        size_bytes = get_file_size(path)
    except Exception:
        size_bytes = 0
    if size_bytes and size_bytes > max_mb * 1024 * 1024:
        return ""
    docs = load_documents(path)
    if not docs:
        return ""
    text = " ".join(d.page_content for d in docs)
    return text[:max_chars]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


def _scaled_similarity(sim: float) -> float:
    return (sim + 1.0) / 2.0


def _keyword_score(tokens: List[str], keyword_weights: Dict[str, float]) -> float:
    if not keyword_weights:
        return 0.0
    hits = 0.0
    total = 0.0
    token_set = set(tokens)
    for kw, weight in keyword_weights.items():
        total += weight
        if kw in token_set:
            hits += weight
    if total == 0.0:
        return 0.0
    return hits / total


def _embed_in_batches(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    if not texts:
        return []
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        embeddings.extend(embed_texts(chunk, batch_size=batch_size))
    return embeddings


def _render_llm_prompt(
    target_labels: List[str],
    filename: str,
    parent_parts: List[str],
    content_excerpt: str,
) -> str:
    parents = " / ".join(parent_parts) if parent_parts else "(none)"
    excerpt = content_excerpt.replace("\n", " ").strip()
    if len(excerpt) > 800:
        excerpt = excerpt[:800] + "..."
    targets = "\n".join(f"- {label}" for label in target_labels)
    return (
        "You classify files into exactly one target category from the list below.\n"
        "Return ONLY the exact target label.\n\n"
        f"Targets:\n{targets}\n\n"
        f"Filename: {filename}\n"
        f"Parent folders: {parents}\n"
        f"Content excerpt: {excerpt or '(empty)'}\n"
    )


def _match_target_label(response: str, target_labels: List[str]) -> str:
    if not response:
        return ""
    cleaned = response.strip().strip('"').strip("'")
    for label in target_labels:
        if cleaned.lower() == label.lower():
            return label
    # Try simple containment as fallback
    for label in target_labels:
        if label.lower() in cleaned.lower():
            return label
    return ""


def build_sort_plan(options: SortOptions) -> List[SortPlanItem]:
    targets = _list_topic_targets(options.root, options.alias_map_text)
    if not targets:
        return []

    exclude_roots = {t.path for t in targets}
    for target in targets:
        parent = Path(target.path).parent
        if _TOPIC_PATTERN.match(parent.name):
            exclude_roots.add(str(parent))
    files = _scan_files(options.root, exclude_roots, options.max_files)
    if not files:
        return []

    logger.info("Smart sort scan: %d files, %d targets", len(files), len(targets))

    meta_texts: List[str] = []
    file_tokens: List[List[str]] = []
    content_texts: List[str] = []
    for path in files:
        meta_text, tokens = _build_meta_text(path, options.root, options.max_parent_levels)
        meta_texts.append(meta_text)
        file_tokens.append(tokens)
        if options.include_content:
            content_texts.append(
                _load_content_text(path, options.max_content_chars, options.max_content_mb)
            )
        else:
            content_texts.append("")

    target_texts = [f"{t.label} {' '.join(t.keywords.keys())}".strip() for t in targets]
    target_embeddings = _embed_in_batches(target_texts)
    meta_embeddings = _embed_in_batches(meta_texts)

    content_embeddings: List[List[float]] = []
    if options.include_content:
        content_embeddings = _embed_in_batches(
            [t if t else "empty" for t in content_texts]
        )
    else:
        content_embeddings = [[] for _ in files]

    total_weight = options.weight_meta + options.weight_content + options.weight_keyword
    if total_weight <= 0:
        total_weight = 1.0
    weight_meta = options.weight_meta / total_weight
    weight_content = options.weight_content / total_weight
    weight_keyword = options.weight_keyword / total_weight

    plan: List[SortPlanItem] = []
    for idx, path in enumerate(files):
        best: Optional[SortPlanItem] = None
        second_best: Optional[SortPlanItem] = None
        for t_idx, target in enumerate(targets):
            meta_sim = _scaled_similarity(_cosine_similarity(meta_embeddings[idx], target_embeddings[t_idx]))
            content_sim = 0.0
            if options.include_content and content_embeddings:
                content_sim = _scaled_similarity(
                    _cosine_similarity(content_embeddings[idx], target_embeddings[t_idx])
                )
            key_score = _keyword_score(file_tokens[idx], target.keywords)
            confidence = (
                (weight_meta * meta_sim)
                + (weight_content * content_sim)
                + (weight_keyword * key_score)
            )
            reason = (
                f"meta={meta_sim:.2f}; content={content_sim:.2f}; keywords={key_score:.2f}"
            )
            candidate = SortPlanItem(
                path=normalize_path(path),
                target_label=target.label,
                target_path=normalize_path(target.path),
                confidence=confidence,
                meta_similarity=meta_sim,
                content_similarity=content_sim,
                keyword_score=key_score,
                reason=reason,
            )
            if best is None or candidate.confidence > best.confidence:
                second_best = best
                best = candidate
            elif second_best is None or candidate.confidence > second_best.confidence:
                second_best = candidate
        if best:
            if second_best is not None:
                best.second_confidence = second_best.confidence
                best.top2_margin = best.confidence - second_best.confidence
            plan.append(best)

    if options.use_llm_fallback:
        llm_status = check_llm_status()
        if not llm_status.get("active"):
            logger.warning("LLM fallback requested but LLM is not active.")
            return plan

        target_labels = [t.label for t in targets]
        remaining = options.llm_max_items
        for idx, item in enumerate(plan):
            if remaining <= 0:
                break
            if item.confidence >= options.llm_confidence_floor:
                continue
            if not content_texts[idx] and not meta_texts[idx]:
                continue
            filename = os.path.basename(item.path)
            rel = os.path.relpath(item.path, options.root)
            parent_parts = list(Path(rel).parts[:-1])
            prompt = _render_llm_prompt(
                target_labels=target_labels,
                filename=filename,
                parent_parts=parent_parts[-options.max_parent_levels :],
                content_excerpt=content_texts[idx],
            )
            response = ask_llm(
                prompt,
                mode="completion",
                model=llm_status.get("current_model"),
                max_tokens=32,
                temperature=0.0,
            )
            match = _match_target_label(response, target_labels)
            if not match:
                continue
            item.target_label = match
            for target in targets:
                if target.label == match:
                    item.target_path = normalize_path(target.path)
                    break
            item.confidence = max(item.confidence, options.llm_confidence_floor)
            item.reason = f"llm={match}; {item.reason}"
            remaining -= 1

    return plan


def _resolve_collision(dest_dir: str, filename: str) -> str:
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        alt = os.path.join(dest_dir, f"{stem} ({counter}){suffix}")
        if not os.path.exists(alt):
            return alt
        counter += 1


def apply_sort_plan(plan: List[SortPlanItem], min_confidence: float, dry_run: bool = True) -> Dict[str, object]:
    moved: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for item in plan:
        if item.confidence < min_confidence:
            skipped.append({"path": item.path, "reason": "below_threshold"})
            continue
        dest_dir = item.target_path
        try:
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
            dest_path = _resolve_collision(dest_dir, os.path.basename(item.path))
            if dry_run:
                moved.append({"from": item.path, "to": normalize_path(dest_path)})
            else:
                shutil.move(item.path, dest_path)
                moved.append({"from": item.path, "to": normalize_path(dest_path)})
        except Exception as exc:
            errors.append({"path": item.path, "error": str(exc)})

    return {"moved": moved, "skipped": skipped, "errors": errors}
