"""Tests for the options scheduled benchmark runs use: a spending cap, a minimum
score, and a Markdown summary (eval_answers.py, shared with eval_xbrl.py).

The questions are answered by a stub, so nothing is called.
"""

import pytest

from financial_rag import eval_answers
from financial_rag.eval_answers import Budget, Case


def _row(question, ok=True, usd=0.01):
    return {"kind": "calculation", "question": question, "asked_as": question,
            "answer": "**FY2024: 44.6%** (msft.htm, p.56)" if ok else "wrong",
            "route": "calculation", "correct": ok, "route_ok": True, "verified": True,
            "cited": ok, "seconds": 0.1, "tokens_in": 100, "tokens_out": 10, "usd": usd,
            "model": "gpt-4o-mini"}


@pytest.fixture
def three_questions(monkeypatch, tmp_path):
    cases = [Case("calculation", f"Question {n}?", ["44.6%"], "calculation") for n in range(3)]
    monkeypatch.setattr(eval_answers, "CASES", cases)
    monkeypatch.setattr(eval_answers, "RESULTS_PATH", tmp_path / "results.json")
    asked = []

    def answer(case, wrong=()):
        asked.append(case.question)
        return _row(case.question, ok=case.question not in wrong)

    monkeypatch.setattr(eval_answers, "run", answer)
    return asked


def test_a_passing_run_exits_0_and_writes_a_summary(three_questions, tmp_path):
    summary = tmp_path / "summary.md"
    assert eval_answers.main(["--min-score", "0.9", "--summary", str(summary)]) == 0
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("### Answers benchmark: 3/3 passed")
    assert "| calculation | 3 | 3 | 3 | 3 | 3 |" in text
    assert "Not passed" not in text


def test_a_score_below_the_minimum_exits_1(three_questions, monkeypatch, tmp_path):
    monkeypatch.setattr(eval_answers, "run",
                        lambda case: _row(case.question, ok=case.question != "Question 1?"))
    summary = tmp_path / "summary.md"
    assert eval_answers.main(["--min-score", "0.9", "--summary", str(summary)]) == \
        eval_answers.EXIT_BELOW_SCORE
    assert "- Question 1? (route calculation)" in summary.read_text(encoding="utf-8")
    # Without a minimum, a miss is reported but not fatal
    assert eval_answers.main([]) == 0


def test_the_spending_cap_stops_the_run(three_questions):
    assert eval_answers.main(["--max-cost", "0.02"]) == eval_answers.EXIT_OVER_BUDGET
    assert three_questions == ["Question 0?", "Question 1?"]    # $0.01 each


def test_a_cap_refuses_a_model_without_a_price():
    budget = Budget(1.0)
    budget.charge(0.5)
    assert not budget.exhausted
    with pytest.raises(SystemExit, match="needs a price"):
        budget.charge(None)
    uncapped = Budget(None)
    uncapped.charge(None)
    assert not uncapped.exhausted
