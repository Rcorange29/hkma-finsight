"""Model names are set in config.py only, so a model swap is one change."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# eval_answers.py keys its price table by model name; config.py is the source
ALLOWED = {"config.py", "eval_answers.py"}
MODEL_LITERAL = re.compile(r"""["'](?:gpt-|text-embedding-|o\d-|cross-encoder/)""")


def test_no_model_names_outside_config():
    offenders = [
        f"{path.name}:{n}"
        for path in sorted(ROOT.glob("*.py")) if path.name not in ALLOWED
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if MODEL_LITERAL.search(line)
    ]
    assert offenders == []
