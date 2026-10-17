"""Keep the corrected review separate from the immutable historical experiment."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = json.loads((ROOT / 'validation/negative_cases_reviewed.json').read_text())


def test_every_original_negative_has_one_review_with_unchanged_question():
    with (ROOT / 'validation/threshold_per_question.csv').open() as stream:
        original = {r['qid']: r['question'] for r in csv.DictReader(stream)
                    if r['expected_refuse'] == 'True'}
    cases = REVIEW['cases']
    assert len(cases) == len({c['qid'] for c in cases}) == 19
    assert {c['qid']: c['question'] for c in cases} == original


def test_raw_experiments_remain_byte_identical():
    for path, digest in REVIEW['legacy_files_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_context_only_pages_cannot_be_mislabeled_as_answer_support():
    for case in REVIEW['cases']:
        assert case['corpus_label'] in REVIEW['label_definitions']
        supporting = sorted({e['pdf_page'] for e in case['evidence']
                             if e['role'] == 'answer_support'})
        assert case['answer_support_pages'] == supporting
        assert all(1 <= p <= 334 for p in case['reviewed_pdf_pages'])
        if case['corpus_label'] == 'sufficient':
            assert supporting
        if case['corpus_label'] == 'insufficient':
            assert not supporting
        assert case['generation_evaluated'] is False


def test_salary_is_answerable_but_undated_headcount_needs_clarification():
    cases = {c['qid']: c for c in REVIEW['cases']}
    assert cases['N14']['corpus_label'] == 'sufficient'
    assert cases['N14']['answer_support_pages'] == [201]
    assert cases['N16']['corpus_label'] == 'ambiguous'
    assert 200 in cases['N16']['reviewed_pdf_pages']
