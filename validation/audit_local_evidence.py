"""Read-only audit of existing local artifacts; no models, network or paid calls.

Run from the repository root: .venv/bin/python validation/audit_local_evidence.py
Requires the existing PDF and Chroma SQLite file. Synthetic unit tests live in tests/.
"""
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
pdf = ROOT / "data/AR2024_E.pdf"
reader = PdfReader(pdf)
with sqlite3.connect(f"file:{ROOT / 'chroma_db/chroma.sqlite3'}?mode=ro", uri=True) as db:
    rows = db.execute("SELECT id, key, string_value, int_value FROM embedding_metadata").fetchall()
chunks = {}
for identifier, key, string, integer in rows:
    chunks.setdefault(identifier, {})[key] = string if string is not None else integer


def normalized(text):
    return re.sub(r"\s+", "", text)


pages = {p: normalized(reader.pages[p].extract_text()) for p in {d['page'] for d in chunks.values()}}
mismatches = [identifier for identifier, d in chunks.items()
              if normalized(d['chroma:document']) not in pages[d['page']]]
retrieval = list(csv.DictReader((ROOT / 'validation/retrieval_per_question.csv').open()))
threshold = list(csv.DictReader((ROOT / 'validation/threshold_per_question.csv').open()))
salary_text = reader.pages[200].extract_text()
result = {
    'pdf_bytes': pdf.stat().st_size,
    'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
    'pdf_pages': len(reader.pages),
    'chunks': len(chunks),
    'source_files': sorted({d['source_file'] for d in chunks.values()}),
    'owners': sorted({d['owner'] for d in chunks.values()}),
    'companies': sorted({d['company'] for d in chunks.values()}),
    'chunks_not_found_on_recorded_page_ignoring_whitespace': mismatches,
    'page_checks': {str(p): term.lower() in reader.pages[p - 1].extract_text().lower()
                    for p, term in [(71, 'Eddie'), (184, '218.8'),
                                    (211, 'Income and Expenditure Account'),
                                    (216, 'Changes in Equity')]},
    'salary_table_on_pdf_page_201': all(term in salary_text for term in
                                     ['Table 5', 'Fixed pay', '7,361', '2,172', '980']),
    'retrieval_rows': len(retrieval),
    'csv_metrics_not_new_model_runs': {
        lang: {key: sum(int(row[key]) for row in retrieval if row['lang'] == lang)
               for key in ['bge_sem_recall5', 'bge_sem_recall20', 'bge_final_recall5',
                           'multi_sem_recall5', 'multi_sem_recall20', 'multi_final_recall5']}
        for lang in sorted({row['lang'] for row in retrieval})},
    'threshold_rows': len(threshold),
    'admitted_negatives_using_original_unreliable_labels': sum(
        row['expected_refuse'] == 'True' and row['actual_refused'] == 'False'
        for row in threshold),
    'label_correction': 'N14 is not an evidence-absent question: PDF p.201 Table 5 reports CEO pay components.',
}
print(json.dumps(result, indent=2))
assert not mismatches
assert all(result['page_checks'].values())
assert result['salary_table_on_pdf_page_201']
