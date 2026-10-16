"""How the real model rewrites follow-up questions. Opt-in: it calls OpenAI.

    RUN_LIVE_TESTS=1 pytest tests/test_followup_live.py

Checks are on the words that must (or must not) survive the rewrite, since the
exact wording varies. Re-run after changing followup.REWRITE_PROMPT.
"""

import os

import pytest

from financial_rag import followup

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1", reason="calls OpenAI; set RUN_LIVE_TESTS=1"
)

NET_INCOME = [
    ("user", "What was Microsoft's net income in fiscal 2024?"),
    ("assistant", "**Net income** · microsoft-sec-10k.pdf\n- **FY2024: $88,136 million** "
                  "(microsoft-sec-10k.pdf, p.67)"),
]
REVENUE = [
    ("user", "What was total revenue in fiscal 2024?"),
    ("assistant", "Total revenue was $245,122 million (microsoft-sec-10k.pdf, p.67)."),
]

# (history, follow-up, words that must appear, words that must not)
CASES = {
    "another year": (NET_INCOME, "And in 2023?", ["net income", "2023", "microsoft"], ["2024"]),
    "another metric": (REVENUE, "What about operating margin?",
                       ["operating margin", "2024"], ["revenue"]),
    "another company": (NET_INCOME, "What about Acme?", ["acme", "net income"], ["microsoft"]),
    "a pronoun": (REVENUE, "How did it change from the year before?",
                  ["revenue"], []),
    "already standalone": (REVENUE, "Who is the company's auditor?", ["auditor"], ["revenue"]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_follow_ups_are_rewritten_to_stand_alone(case):
    history, question, present, absent = CASES[case]
    rewritten = followup.standalone_question(question, history).lower()
    for word in present:
        assert word in rewritten, rewritten
    for word in absent:
        assert word not in rewritten, rewritten
