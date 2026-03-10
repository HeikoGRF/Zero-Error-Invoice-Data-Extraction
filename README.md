# Zero-Error-Invoice-Data-Extraction

## How to run this app on another machine

### 1. Clone and set up Python

```bash
git clone https://github.com/HeikoGRF/Zero-Error-Invoice-Data-Extraction.git
cd Zero-Error-Invoice-Data-Extraction

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requirements:

- Python 3.11 or 3.12 (recommended)
- Tesseract OCR installed and on your `PATH`
- Poppler installed (for `pdf2image` to render PDFs)

On macOS with Homebrew for example:

```bash
brew install tesseract poppler
```

### 2. Configure environment variables

Create a local `.env` file (not committed) with:

```bash
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3-flash      # or another Gemini text model
OCR_PSM=3                        # Tesseract page segmentation mode (3 works well for invoices)
```

These are loaded automatically by the app via `python-dotenv`.

### 3. Start the API + web UI

From the project root (with venv activated):

```bash
uvicorn app.main:app --reload
```

Open the browser at:

- `http://127.0.0.1:8000/`

The page lets you:

- Upload one or more PDF/image invoices.
- It normalizes them to images, runs OCR, LLM extraction, validation, and shows:
  - extraction/validation status per file,
  - OCR quality metrics (avg confidence, low-confidence ratio),
  - counts of validation errors/warnings.

The structured output for each invoice is also written to disk:

- `storage/ingested/<invoice_id>/page-*.png` (normalized pages)
- `storage/ingested/<invoice_id>/extraction.json` (final structured result including validation and quality)
- `storage/ingested/<invoice_id>/extraction_raw.txt` (raw Gemini output for debugging)

### 4. Run automated validation tests

This repository includes a small fixture-based test suite for the validation engine.

```bash
pytest -q
```

Covered fixture set:

- clean single-page invoice
- multi-page invoice
- noisy scan invoice
- tricky inconsistent totals invoice (deliberate mismatch that must be flagged)

---

## Solution description for the Zero-Error Invoice Extraction challenge

The goal of this repository is to implement a robust, end-to-end pipeline for extracting structured invoice data with a near-zero silent failure rate, as described in the technical challenge PDF. Below is how the current implementation addresses each part of the challenge.

### 1. Initial data processing & ingestion

- **Accepted inputs**: PDF and image invoices (`application/pdf`, `image/png`, `image/jpeg`).
- **Normalization to images**:
  - PDFs are rendered to page images via `pdf2image.convert_from_bytes`.
  - All images (originals and rendered pages) are preprocessed with:
    - grayscale conversion,
    - auto-contrast,
    - mild sharpening,
    - mild upscaling for very small images.
  - Output is one PNG per page: `storage/ingested/<invoice_id>/page-1.png`, `page-2.png`, ...
- **Web UI**:
  - A simple FastAPI-backed upload page (`GET /`) lets a user drag/drop or select invoices.
  - `POST /ingest` handles ingestion and immediately triggers extraction and validation.

This ensures all downstream components see a consistent representation (preprocessed page images), regardless of original format or source.

### 2. OCR & layout preservation

- **OCR engine**: Tesseract via `pytesseract.image_to_data`.
  - Configurable page segmentation mode via `OCR_PSM` (default `3`), which worked better for mixed invoice layouts than mode `6` on degraded scans.
- **Layout reconstruction**:
  - For each page, we produce:
    - `OcrToken` (word-level token) with:
      - `text`
      - normalized bounding box `bbox = (x_min, y_min, x_max, y_max)` in \[0,1\]
      - `confidence`
      - `line_id`, `block_id`
      - `token_index` (position within line)
    - `OcrLine` objects:
      - ordered by `reading_order_index`
      - contain ordered tokens and a line-level bounding box.
    - `OcrPage` objects collecting all lines.
- **Preserved layout**:
  - The OCR endpoint (`GET /ocr/{invoice_id}`) exposes the layout-preserving representation:
    - `pages[].lines[].tokens[]` with full positional and confidence info.
  - This is what the LLM consumes, not the raw image or PDF.

### 3. Data extraction strategy using LLMs

- **LLM role**:
  - The LLM is used as a **semantic interpreter** over layout-preserving OCR text.
  - It is *not* trusted as ground truth; it proposes candidates which are then strictly validated.
- **Prompt structure** (`_build_llm_extraction_prompt`):
  - We pass OCR lines in reading order as:
    - `"[page=1 line_id=p1_b1_l1] ACME Supplies GmbH"`, etc.
  - We provide an explicit JSON schema the model must return:
    - `line_items[]`: `description`, `quantity`, `unit_price`, `line_total`, `discount`, `extra_cost`, `source_line_ids`.
    - `summary`: `subtotal`, `total`, `currency`, `tax_bases[]`, `tax_rates[]`.
    - `metadata`: `invoice_number`, `invoice_date`, `vendor_name`, `customer_name`.
  - Instructions emphasize:
    - Only use values literally present in the text,
    - Use `null` when uncertain,
    - For line items, fill `source_line_ids`,
    - For vendor/customer names, copy exact OCR line text when identifiable.
  - When layout-based OCR extraction finds vendor/customer lines, those values override the LLM output.
- **Model & orchestration**:
  - Using Gemini via `google-generativeai` (`GEMINI_API_KEY`, `GEMINI_MODEL`).
  - `_run_extraction_pipeline(invoice_id)`:
    1. Runs OCR.
    2. Builds the prompt.
    3. Calls Gemini.
    4. Parses JSON robustly (also handles fenced ` ```json ... ``` ` blocks).
    5. Maps into strongly typed Pydantic models (`InvoiceExtraction`, `InvoiceLineItem`, etc.).

In short, the LLM is constrained by a strict schema and instructed to be conservative; it proposes a structured view which we subsequently verify.

### 4. Zero-hallucination validation & correction engine

The core of the challenge is ensuring we do **not** silently accept hallucinated or inconsistent values.

#### 4.1 Token-level provenance

- For each numeric field in every line item (`quantity`, `unit_price`, `line_total`, `discount`, `extra_cost`), we:
  - Build a token index (`_build_token_index`) from `OcrPage` (line → tokens).
  - Find tokens whose text matches the numeric value (`_token_refs_for_value`).
  - Store them as `TokenRef` objects in `InvoiceLineItem.source_tokens[field_name]`, containing:
    - `line_id`
    - `token_index`
    - `text`
    - `bbox`

If we cannot find corresponding tokens, that value is considered suspect.

#### 4.2 Arithmetic + consistency checks

- Implemented in `_validate_extraction`:
  - **Line-item arithmetic**:
    - For each item with `quantity`, `unit_price`, and `line_total` present:
      - compute `expected = quantity * unit_price - discount + extra_cost`.
      - compare to `line_total` with small tolerance; if mismatched, add `line_total_mismatch` error.
  - **Subtotal vs sum of line totals**:
    - `sum(line_total)` vs `summary.subtotal`; mismatches produce `subtotal_mismatch` error.
  - **Total vs subtotal + tax**:
    - Compute tax = Σ(`tax_base[i] * tax_rate[i] / 100`).
    - Check `subtotal + computed_tax` vs `summary.total`; mismatches -> `total_mismatch` error.

#### 4.3 OCR-grounding checks

- Numeric provenance:
  - For each numeric line-item field:
    - If `source_tokens[field]` is empty, we add `missing_token_provenance` error.
  - This guarantees numeric values can be linked back to specific OCR tokens (no naked hallucinations).
- Text metadata:
  - We compute normalized OCR line texts and:
    - If `vendor_name` is not found in the OCR text → `vendor_not_in_ocr` error.
    - If `customer_name` is not found → `customer_not_in_ocr` warning.
  - **Layout-based metadata extraction** (no hardcoded string cleanup):
    - `vendor_name`: first line of block 1 (header) from OCR, or first line before invoice/bill markers if block 1 is not identifiable.
    - `customer_name`: first line after the `"Bill To"` marker.
    - Uses raw OCR output; one OCR line per entity. Correctness depends on OCR line segmentation (PSM 3).

#### 4.4 OCR quality metrics

- `_compute_ocr_quality` aggregates:
  - `page_count`
  - `token_count`
  - `avg_token_confidence`
  - `low_confidence_ratio` (fraction of tokens below a confidence threshold).
- These are stored in `InvoiceExtraction.quality` and used to add validation warnings:
  - `low_avg_ocr_confidence`
  - `high_low_confidence_ratio`

#### 4.5 Validation output

- `InvoiceValidation` model:
  - `status`: `"ok" | "partial" | "failed"`
  - `issues`: list of `ValidationIssue` with:
    - `level`: `"warning"` or `"error"`
    - `code`: e.g. `line_total_mismatch`, `subtotal_mismatch`, `total_mismatch`, `missing_token_provenance`, `vendor_not_in_ocr`.
    - `message`
    - `path` indicating the field location (e.g. `line_items[1].line_total`).
- The final `extraction.json` includes:
  - `line_items`, `summary`, `metadata`
  - `quality`
  - `validation`

Any serious inconsistency leads to `validation.status = "failed"`, and consumers can treat such invoices as requiring manual review rather than trusted automation.

### 5. API endpoints and UI exposure

- **`POST /ingest`**:
  - Accepts files and:
    - normalizes them to images,
    - runs the extraction pipeline,
    - persists `extraction.json` / `extraction_raw.txt`,
    - returns per-file:
      - `invoice_id`
      - `extraction_created` / `extraction_error`
      - `quality`
      - `validation`.
- **`GET /extract/{invoice_id}`**:
  - Re-runs extraction/validation for an existing invoice.
- **`GET /analysis/{invoice_id}`**:
  - Returns a compact view containing only:
    - `invoice_id`
    - `quality`
    - `validation`.
- **Web UI summary**:
  - After upload, the UI displays:
    - files uploaded vs extraction failures,
    - validation status counts,
    - average OCR confidence,
    - counts of validation errors/warnings,
    - per-file extraction failure notes.

This makes the health of each extraction immediately visible to the user, not hidden in logs.

### 6. Automated tests & sample set

- Location: `tests/fixtures/invoices/*.json` and `tests/test_validation_fixtures.py`.
- Fixtures:
  - `clean_single_page.json`
  - `multi_page.json`
  - `noisy_scan.json`
  - `tricky_totals_inconsistent.json` (deliberately incorrect subtotal/total).
- Tests verify that:
  - clean/multi-page/noisy fixtures produce `validation.status == "ok"` with no issues,
  - the inconsistent-totals fixture yields `validation.status == "failed"` and includes:
    - `subtotal_mismatch`
    - `total_mismatch`.

This ensures the zero-hallucination engine does not silently accept wrong totals, and provides a basis for regression testing as the system evolves.

---

In summary, this codebase implements the challenge requirements with:

- a robust ingestion and normalization step,
- layout-aware OCR,
- schema-constrained LLM extraction used only as a candidate generator,
- **layout-based metadata extraction** for vendor/customer (one OCR line per entity, no hardcoded string cleanup),
- a dedicated zero-hallucination engine that re-computes and validates all critical values,
- explicit provenance to OCR tokens,
- surfaced quality/validation in both API and UI,
- and automated tests around both clean and adversarial invoice scenarios.

