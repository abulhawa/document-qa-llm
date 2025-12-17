from pathlib import Path
import ast


def _core_python_files():
    core_dir = Path(__file__).resolve().parent.parent / "core"
    return [p for p in core_dir.rglob("*.py") if p.is_file()]


def test_core_has_no_qa_pipeline_imports():
    offending = []
    for path in _core_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("qa_pipeline"):
                        offending.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("qa_pipeline"):
                    offending.append(f"{path}: from {module} import ...")

    assert not offending, "core/ must not import qa_pipeline: " + "; ".join(offending)
