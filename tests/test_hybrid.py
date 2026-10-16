"""Tests for the keyword half of hybrid retrieval (hybrid.py). Pure functions only."""

from collections import Counter

import pytest

from financial_rag.hybrid import (
    KEYWORD_WEIGHT_AMOUNT,
    KEYWORD_WEIGHT_DEFAULT,
    KEYWORD_WEIGHT_YEAR,
    RRF_K,
    Indexed,
    bm25_scores,
    focus,
    has_amounts,
    index_text,
    profile_query,
    rrf,
    tokenize,
)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        # Grouping commas and $ don't matter: the filing's "$ 245,122" is the user's "245122"
        ("Total revenue $ 245,122", ["total", "revenue", "245122"]),
        ("245122", ["245122"]),
        # Parenthesised negatives and per-share decimals stay whole
        ("Other income (1,646) and EPS $11.80", ["income", "1646", "eps", "11.80"]),
        ("grew 15.7%", ["grew", "15.7"]),
        # Sentence-ending punctuation isn't part of the number
        ("was 245,122. Then", ["245122"]),
    ],
)
def test_numbers_are_matched_however_they_are_written(text, tokens):
    assert [t for t in tokenize(text) if t in tokens] == tokens


@pytest.mark.parametrize("question", ["FY24 revenue", "FY2024 revenue", "fy 24 revenue",
                                      "FY'24 revenue", "fiscal 2024 revenue"])
def test_fiscal_year_abbreviations_match_the_filings_wording(question):
    assert tokenize(question) == ["fiscal", "2024", "revenue"]


def test_plurals_fold_and_stopwords_drop():
    assert tokenize("What were the revenues and liabilities?") == ["revenue", "liability"]
    assert tokenize("income taxes") == tokenize("income tax")
    # Words that only look plural are left alone
    assert tokenize("gross business analysis") == ["gross", "business", "analysis"]


def test_possessives_and_single_letters_drop():
    assert tokenize("the company's 10-K") == ["company", "10"]


# ---------------------------------------------------------------------------
# Query profile
# ---------------------------------------------------------------------------

def test_quoted_values_weight_the_keyword_ranking_most():
    p = profile_query("Which line was $57,582 million in fiscal 2024?")
    assert p.amounts == ("57582",)
    assert p.years == ("2024",)
    assert p.keyword_weight == KEYWORD_WEIGHT_AMOUNT


def test_a_named_year_alone_gets_the_year_weight():
    p = profile_query("What was net income for FY23?")
    assert (p.amounts, p.years) == ((), ("2023",))
    assert p.keyword_weight == KEYWORD_WEIGHT_YEAR


def test_conceptual_questions_lean_on_semantic_search():
    assert profile_query("What are the main risk factors?").keyword_weight == KEYWORD_WEIGHT_DEFAULT


def test_small_numbers_and_dates_are_not_amounts():
    p = profile_query("Top 3 risks as of June 30, 2024")
    assert p.amounts == ()
    assert p.keyword_weight == KEYWORD_WEIGHT_YEAR


def test_query_terms_are_distinct_and_in_order():
    assert profile_query("revenue revenue growth revenue").terms == ("revenue", "growth")


def test_has_amounts_needs_every_quoted_value():
    p = profile_query("Was it 57,582 or 53,813?")
    assert has_amounts(p, index_text("Unearned revenue 57,582 53,813"))
    assert not has_amounts(p, index_text("Unearned revenue 57,582"))
    # No quoted values: nothing gets priority
    assert not has_amounts(profile_query("unearned revenue"), index_text("Unearned revenue 57,582"))


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _docs(*texts):
    return [index_text(t) for t in texts]


def test_exact_value_ranks_its_chunk_first():
    docs = _docs(
        "Revenue increased 16% driven by growth across each segment",
        "Short-term unearned revenue 57,582 53,813",
        "Total revenue 245,122 211,915 198,270",
    )
    scores = bm25_scores(profile_query("unearned revenue 57,582").terms, docs)
    assert scores.index(max(scores)) == 1


def test_rare_terms_outweigh_common_ones():
    docs = _docs("revenue goodwill", "revenue", "revenue", "revenue")
    scores = bm25_scores(["revenue", "goodwill"], docs)
    assert scores[0] > scores[1] > 0


def test_no_match_scores_zero():
    assert bm25_scores(["goodwill"], _docs("revenue", "net income")) == [0.0, 0.0]


def test_empty_inputs():
    assert bm25_scores([], _docs("revenue")) == [0.0]
    assert bm25_scores(["revenue"], []) == []


def test_longer_chunks_are_normalised():
    short = Indexed(Counter({"goodwill": 1}), 10)
    long = Indexed(Counter({"goodwill": 1}), 1000)
    s_short, s_long = bm25_scores(["goodwill"], [short, long])
    assert s_short > s_long


# ---------------------------------------------------------------------------
# Fusion and excerpts
# ---------------------------------------------------------------------------

def test_rrf_rewards_chunks_both_searches_found():
    fused = dict(rrf([(["a", "b", "c"], 1.0), (["c", "d"], 1.0)]))
    assert fused["c"] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert max(fused, key=fused.get) == "c"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_weight_shifts_the_order():
    semantic, keyword = ["s1", "s2"], ["k1", "k2"]
    assert rrf([(semantic, 1.0), (keyword, 2.0)])[0][0] == "k1"
    assert rrf([(semantic, 1.0), (keyword, 0.5)])[0][0] == "s1"


def test_focus_shows_the_part_of_a_long_chunk_that_matches():
    text = " ".join(["filler"] * 200 + ["Goodwill", "119,220"] + ["filler"] * 200)
    excerpt = focus(text, ("goodwill", "119220"), max_words=20)
    assert "Goodwill 119,220" in excerpt
    assert len(excerpt.split()) == 20


def test_focus_leaves_short_or_unmatched_text_alone():
    assert focus("Total revenue 245,122", ("revenue",)) == "Total revenue 245,122"
    long = " ".join(["filler"] * 100)
    assert focus(long, ("goodwill",), max_words=20) == long
