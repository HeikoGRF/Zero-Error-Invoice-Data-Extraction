from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.main import InvoiceExtraction, _validate_extraction


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "invoices"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "clean_single_page.json",
        "multi_page.json",
        "noisy_scan.json",
    ],
)
def test_cleanish_samples_validate_ok(fixture_name: str) -> None:
    fixture = _load_fixture(fixture_name)
    extraction = InvoiceExtraction.model_validate(fixture["extraction"])

    validation = _validate_extraction(
        extraction.invoice_id,
        extraction,
        ocr_numbers_by_line={},
        ocr_lines=fixture["ocr_lines"],
    )

    assert validation.status == fixture["expect"]["status"]
    assert validation.issues == []


def test_inconsistent_totals_are_flagged() -> None:
    fixture = _load_fixture("tricky_totals_inconsistent.json")
    extraction = InvoiceExtraction.model_validate(fixture["extraction"])

    validation = _validate_extraction(
        extraction.invoice_id,
        extraction,
        ocr_numbers_by_line={},
        ocr_lines=fixture["ocr_lines"],
    )

    assert validation.status == fixture["expect"]["status"]
    issue_codes = {issue.code for issue in validation.issues}
    for required_code in fixture["expect"]["required_issue_codes"]:
        assert required_code in issue_codes

