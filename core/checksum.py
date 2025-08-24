from typing import Union
import hashlib
from utils.file_utils import compute_checksum as file_checksum


def chunk_id(checksum: Union[str, bytes], idx: int) -> str:
    if isinstance(checksum, bytes):
        checksum = checksum.decode("utf-8", errors="ignore")
    base = f"{checksum}:{idx}".encode("utf-8")
    return hashlib.md5(base).hexdigest()
