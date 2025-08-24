import os

ALLOW_FILE_DELETE = os.getenv("ALLOW_FILE_DELETE", "false").lower() == "true"

def require_delete_enabled():
    if not ALLOW_FILE_DELETE:
        raise PermissionError("File deletion is disabled by configuration.")
