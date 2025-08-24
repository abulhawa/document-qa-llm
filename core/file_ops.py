import os
from core.safety import require_delete_enabled


def delete_file(path: str) -> None:
    require_delete_enabled()
    os.remove(path)
