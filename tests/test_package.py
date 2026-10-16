"""Every module imports its siblings through the package, even inside functions.

A bare `import ingest` works only when the repository root is the working
directory's script path; under `python -m financial_rag.<module>` (and in the
Docker image) it fails at the moment that code runs.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "financial_rag"


def test_no_bare_imports_of_package_modules():
    modules = {p.stem for p in PACKAGE.glob("*.py")} - {"__init__"}
    assert len(modules) > 10
    bare = []
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            bare += [f"{path.name}:{node.lineno} {n}" for n in names if n in modules]
    assert bare == []
