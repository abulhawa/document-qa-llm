import os

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md"}

EXCLUDE_DIRS = [
    r"C:\\Windows",
    r"C:\\Program Files",
    r"C:\\Program Files (x86)",
    r"C:\\ProgramData",
    r"C:\\$Recycle.Bin",
    r"C:\\Recovery",
    r"C:\\Users\\*\\AppData",
    ".git",
    "node_modules",
    "__pycache__",
]

def _norm(p: str) -> str:
    return p.replace("\\", "/").lower()

def should_skip(path: str) -> bool:
    p = _norm(path)
    # quick folder excludes
    for pat in EXCLUDE_DIRS:
        head = _norm(pat).rstrip("*")
        if head and p.startswith(head):
            return True
    base = os.path.basename(p)
    if base.startswith("~$") or base.endswith(".tmp") or base.endswith(".lnk"):
        return True
    return not any(p.endswith(ext) for ext in (e.lower() for e in ALLOWED_EXT))
