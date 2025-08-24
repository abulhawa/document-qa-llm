import os


def _load_map():
    spec = os.getenv("DOC_PATH_MAP", "")
    pairs = []
    for item in filter(None, (p.strip() for p in spec.split(";"))):
        host, cont = map(str.strip, item.split("=>", 1))
        pairs.append((host.replace("\\", "/").rstrip("/"), cont.rstrip("/")))
    return pairs


_PATH_MAP = _load_map()


def to_worker_path(path: str) -> str:
    p = path.replace("\\", "/")
    for host, cont in _PATH_MAP:
        if p.lower().startswith(host.lower()):
            return cont + p[len(host):]
    return p
