import os
from typing import Tuple

def _parse_map(env: str) -> Tuple[Tuple[str, str], ...]:
    pairs = []
    for item in env.split(";"):
        if not item.strip() or "=>" not in item:
            continue
        src, dst = item.split("=>", 1)
        pairs.append((src.rstrip("/\\").lower(), dst.rstrip("/")))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return tuple(pairs)

_PATH_MAP = _parse_map(os.getenv("DOC_PATH_MAP", ""))

def host_to_container_path(host_path: str) -> str:
    hp = host_path.replace("\\", "/")
    low = hp.lower()
    for src, dst in _PATH_MAP:
        if low.startswith(src):
            return dst + hp[len(src):]
    return hp