#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import subprocess


def main() -> int:
    # Ensure e2e-safe test namespace so indices are isolated
    os.environ.setdefault("TEST_MODE", "e2e")
    # Optional: give tests their own namespace if user didn’t set one
    if not os.environ.get("NAMESPACE"):
        os.environ["NAMESPACE"] = "test"

    # Run only e2e tests
    cmd = [sys.executable, "-m", "pytest", "-m", "e2e"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

