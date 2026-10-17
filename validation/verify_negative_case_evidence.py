"""Verify reviewed quotations/search hits against the existing local PDF, offline.

Run: .venv/bin/python validation/verify_negative_case_evidence.py
This verifies provenance, not the semantic correctness of the human labels.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    review = json.loads((ROOT / 'validation/negative_cases_reviewed.json').read_text())
    pdf = ROOT / 'data' / review['source_file']
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == review['source_pdf_sha256']
    reader = PdfReader(pdf)
    assert len(reader.pages) == review['source_pdf_pages']
    pages = [re.sub(r'\s+', ' ', page.extract_text()).strip() for page in reader.pages]
    quotations = 0
    searches = 0
    for case in review['cases']:
        for evidence in case['evidence']:
            assert evidence['excerpt'] in pages[evidence['pdf_page'] - 1], case['qid']
            quotations += 1
        for search in case['full_text_searches']:
            actual = [n for n, text in enumerate(pages, 1)
                      if re.search(search['pattern'], text, re.I)]
            assert actual == search['matched_pdf_pages'], (case['qid'], search['pattern'])
            searches += 1
    for path, expected in review['legacy_files_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    print(json.dumps({'cases': len(review['cases']), 'pdf_pages': len(pages),
                      'verified_excerpts': quotations, 'verified_searches': searches,
                      'labels': dict(Counter(c['corpus_label'] for c in review['cases'])),
                      'legacy_csvs_byte_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
