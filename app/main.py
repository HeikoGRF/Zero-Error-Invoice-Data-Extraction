from __future__ import annotations

from pathlib import Path
import io
import os
import uuid
import re
import json
from typing import Dict, List, Tuple, Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pytesseract import Output
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

ACCEPTED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
}

BASE_STORAGE_DIR = Path("storage") / "ingested"

# Load environment variables from .env if present.
load_dotenv()

# OCR page segmentation mode. 3 is more robust for mixed invoice layouts;
# mode 6 tended to merge many regions into one line on degraded images.
OCR_PSM = os.getenv("OCR_PSM", "3")


def ensure_storage_dir() -> None:
    BASE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Apply light-weight preprocessing to improve OCR robustness:
    - Convert to grayscale (no forced rotation).
    - Auto-contrast to stretch histogram.
    - Mild sharpening.
    - Mild upscaling for very small images.

    Note: Sharpening and upscaling are intentionally conservative so they
    can help OCR on slightly soft scans without aggressively amplifying blur.
    """
    # Convert to grayscale.
    img = img.convert("L")

    # Auto-contrast to enhance text/background separation.
    img = ImageOps.autocontrast(img, cutoff=2)

    # Mild sharpening to crisp up edges.
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=75, threshold=5))

    # Mild upscaling for very low-resolution images.
    width, height = img.size
    min_dim = min(width, height)
    if min_dim < 800:
        scale = 800 / float(min_dim)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.BICUBIC)

    return img


app = FastAPI(title="Zero-Error Invoice Extraction - Ingestion")


def get_gemini_model() -> genai.GenerativeModel:
    """
    Lazily configure and return a Gemini GenerativeModel instance using
    GEMINI_API_KEY and GEMINI_MODEL from the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY in .env "
            "(or GOOGLE_API_KEY as fallback)."
        )
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash")
    return genai.GenerativeModel(model_name)


def _parse_llm_json(raw_text: str) -> dict:
    """
    Parse JSON from model output robustly.
    Gemini may return plain JSON or JSON wrapped in markdown fences.
    """
    import json

    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    # Direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try fenced json block: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Last fallback: first JSON object-like block.
    obj_match = re.search(r"(\{[\s\S]*\})", text)
    if obj_match:
        return json.loads(obj_match.group(1))

    raise ValueError("No valid JSON object found in model response")


def _collect_ocr_numbers_by_line(pages: List[OcrPage]) -> Dict[str, List[str]]:
    """
    Build a mapping line_id -> list of numeric-looking token substrings.
    Used to verify that extracted numeric values actually appear in OCR text.
    """
    line_numbers: Dict[str, List[str]] = {}
    number_pattern = re.compile(r"[0-9][0-9.,]*")

    for page in pages:
        for line in page.lines:
            bucket = line_numbers.setdefault(line.id, [])
            for token in line.tokens:
                for match in number_pattern.findall(token.text):
                    bucket.append(match)

    return line_numbers


def _collect_ocr_line_texts(pages: List[OcrPage]) -> List[str]:
    """
    Collect full OCR line texts in reading order.
    """
    texts: List[str] = []
    for page in pages:
        for line in sorted(page.lines, key=lambda l: l.reading_order_index):
            txt = " ".join(token.text for token in line.tokens).strip()
            if txt:
                texts.append(txt)
    return texts


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _text_in_ocr_lines(value: str | None, ocr_lines: List[str]) -> bool:
    """
    Check whether a text value appears in OCR line texts (case-insensitive).
    """
    if not value:
        return False
    needle = _normalize_text(value)
    return any(needle in _normalize_text(line) for line in ocr_lines)


def _extract_vendor_customer_from_ocr(pages: List[OcrPage]) -> Tuple[str | None, str | None]:
    """
    Deterministic fallback for vendor/customer from OCR layout:
    - vendor: first meaningful line near top of page 1
    - customer: first line after a 'Bill To' marker
    """
    if not pages:
        return None, None

    page1 = pages[0]
    lines = [
        " ".join(token.text for token in line.tokens).strip()
        for line in sorted(page1.lines, key=lambda l: l.reading_order_index)
    ]
    lines = [l for l in lines if l]
    if not lines:
        return None, None

    vendor: str | None = None
    customer: str | None = None

    # Vendor candidate: first line before invoice/bill markers.
    stop_words = ("invoice", "bill to", "due date", "purchase order")
    for line in lines:
        norm = _normalize_text(line)
        if any(w in norm for w in stop_words):
            break
        if len(norm) >= 3:
            vendor = line
            break

    # Customer candidate: line right after "Bill To" marker.
    bill_idx = None
    for i, line in enumerate(lines):
        if "bill to" in _normalize_text(line):
            bill_idx = i
            break
    if bill_idx is not None:
        for candidate in lines[bill_idx + 1 :]:
            if len(_normalize_text(candidate)) >= 3:
                customer = candidate
                break

    return vendor, customer


def _value_appears_in_ocr(
    value: float | int,
    source_line_ids: List[str],
    ocr_numbers_by_line: Dict[str, List[str]],
) -> bool:
    """
    Check if a numeric value occurs in the OCR text for the given lines
    (with simple normalization of decimal separators and formatting).
    """
    if not source_line_ids:
        return False

    # Generate a few common string representations.
    candidates: List[str] = []
    try:
        candidates.append(str(int(value)))
    except (TypeError, ValueError):
        pass

    try:
        candidates.append(f"{float(value):.2f}")
    except (TypeError, ValueError):
        pass

    normalized_candidates = set(c.replace(",", ".") for c in candidates if c)

    for line_id in source_line_ids:
        numbers = ocr_numbers_by_line.get(line_id) or []
        for num in numbers:
            norm = num.replace(",", ".")
            for cand in normalized_candidates:
                if cand and cand in norm:
                    return True

    return False


def _build_token_index(pages: List[OcrPage]) -> Dict[str, List[OcrToken]]:
    """
    Map line_id -> ordered tokens (with token_index set).
    """
    idx: Dict[str, List[OcrToken]] = {}
    for page in pages:
        for line in page.lines:
            idx[line.id] = line.tokens
    return idx


def _token_refs_for_value(
    value: float | int,
    source_line_ids: List[str],
    tokens_by_line: Dict[str, List[OcrToken]],
) -> List[TokenRef]:
    """
    Find OCR tokens (within the provided source_line_ids) that contain the given
    numeric value (best-effort string matching with normalization).
    """
    if value is None or not source_line_ids:
        return []

    candidates: List[str] = []
    try:
        candidates.append(str(int(value)))
    except (TypeError, ValueError):
        pass
    try:
        candidates.append(f"{float(value):.2f}")
    except (TypeError, ValueError):
        pass

    normalized_candidates = [c.replace(",", ".") for c in candidates if c]
    refs: List[TokenRef] = []

    for line_id in source_line_ids:
        for tok in tokens_by_line.get(line_id, []):
            t = tok.text.replace(",", ".")
            if tok.token_index is None:
                continue
            if any(c and c in t for c in normalized_candidates):
                refs.append(
                    TokenRef(
                        line_id=line_id,
                        token_index=tok.token_index,
                        text=tok.text,
                        bbox=tok.bbox,
                    )
                )
    return refs


def _attach_token_provenance(extraction: InvoiceExtraction, pages: List[OcrPage]) -> None:
    """
    Attach token-level provenance for each numeric field in line items.
    This makes LLM outputs auditable and enables strict validation.
    """
    tokens_by_line = _build_token_index(pages)

    for item in extraction.line_items:
        item.source_tokens = {}
        for field_name in ("quantity", "unit_price", "line_total", "discount", "extra_cost"):
            value = getattr(item, field_name)
            if value is None:
                continue
            refs = _token_refs_for_value(value, item.source_line_ids, tokens_by_line)
            item.source_tokens[field_name] = refs


def _compute_ocr_quality(pages: List[OcrPage]) -> OcrQuality:
    """
    Compute simple OCR quality metrics to expose in API/UI.
    """
    confidences: List[float] = []
    for page in pages:
        for line in page.lines:
            for token in line.tokens:
                if token.confidence >= 0:
                    confidences.append(token.confidence)

    token_count = len(confidences)
    if token_count == 0:
        return OcrQuality(
            page_count=len(pages),
            token_count=0,
            avg_token_confidence=0.0,
            low_confidence_ratio=1.0,
        )

    low_count = sum(1 for c in confidences if c < 60.0)
    avg_conf = sum(confidences) / token_count
    low_ratio = low_count / token_count

    return OcrQuality(
        page_count=len(pages),
        token_count=token_count,
        avg_token_confidence=round(avg_conf, 2),
        low_confidence_ratio=round(low_ratio, 4),
    )


class OcrToken(BaseModel):
    text: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    line_id: str
    block_id: str
    token_index: int | None = None


class TokenRef(BaseModel):
    line_id: str
    token_index: int
    text: str
    bbox: Tuple[float, float, float, float]


class OcrLine(BaseModel):
    id: str
    tokens: List[OcrToken]
    bbox: Tuple[float, float, float, float]
    reading_order_index: int


class OcrPage(BaseModel):
    page_number: int
    width: int
    height: int
    lines: List[OcrLine]


class InvoiceLineItem(BaseModel):
    description: str
    quantity: float | None
    unit_price: float | None
    line_total: float | None
    discount: float | None = None
    extra_cost: float | None = None
    source_line_ids: List[str]
    source_tokens: Dict[str, List[TokenRef]] | None = None


class InvoiceSummary(BaseModel):
    subtotal: float | None
    total: float | None
    currency: str | None = None
    tax_bases: List[float] | None = None
    tax_rates: List[float] | None = None


class InvoiceMetadata(BaseModel):
    invoice_number: str | None = None
    invoice_date: str | None = None
    vendor_name: str | None = None
    customer_name: str | None = None


class InvoiceExtraction(BaseModel):
    invoice_id: str
    line_items: List[InvoiceLineItem]
    summary: InvoiceSummary
    metadata: InvoiceMetadata
    raw_model_output: dict
    quality: "OcrQuality | None" = None
    validation: "InvoiceValidation | None" = None


class ValidationIssue(BaseModel):
    level: Literal["warning", "error"]
    code: str
    message: str
    path: str | None = None


class InvoiceValidation(BaseModel):
    status: Literal["ok", "partial", "failed"]
    issues: List[ValidationIssue]


class OcrQuality(BaseModel):
    page_count: int
    token_count: int
    avg_token_confidence: float
    low_confidence_ratio: float


InvoiceExtraction.model_rebuild()


@app.on_event("startup")
async def startup_event() -> None:
    ensure_storage_dir()


@app.get("/", response_class=HTMLResponse)
async def upload_form() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <title>Invoice Uploader</title>
        <style>
          body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background-color: #0f172a;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
          }
          .card {
            background-color: #020617;
            border-radius: 1rem;
            padding: 2rem 2.5rem;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8);
            max-width: 480px;
            width: 100%;
            border: 1px solid rgba(148, 163, 184, 0.25);
          }
          h1 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
          }
          p {
            margin: 0.25rem 0 1.5rem;
            color: #9ca3af;
            font-size: 0.95rem;
          }
          .file-input {
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px dashed rgba(148, 163, 184, 0.6);
            background-color: rgba(15, 23, 42, 0.8);
            margin-bottom: 1rem;
          }
          input[type="file"] {
            width: 100%;
            color: #e5e7eb;
          }
          .hint {
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
          }
          .status {
            margin-top: 1rem;
            font-size: 0.9rem;
            white-space: pre-wrap;
          }
          .status--success {
            color: #4ade80;
          }
          .status--error {
            color: #f97373;
          }
          .summary {
            margin-top: 1rem;
            padding: 0.85rem 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background-color: rgba(15, 23, 42, 0.6);
            font-size: 0.9rem;
          }
          .summary h2 {
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
          }
          .summary-line {
            color: #cbd5e1;
            margin: 0.15rem 0;
          }
          .summary-issues {
            margin-top: 0.55rem;
            color: #fca5a5;
          }
          button {
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            border: none;
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            color: #f9fafb;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.08s ease, box-shadow 0.08s ease, filter 0.08s ease;
          }
          button:hover {
            filter: brightness(1.05);
            box-shadow: 0 16px 30px rgba(56, 189, 248, 0.35);
            transform: translateY(-1px);
          }
          button:active {
            transform: translateY(0);
            box-shadow: none;
          }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Upload Invoice</h1>
          <p>Upload PDF or image invoices to ingest them into the extraction pipeline.</p>
          <form id="upload-form" enctype="multipart/form-data">
            <div class="file-input">
              <input
                type="file"
                name="files"
                multiple
                accept=".pdf,image/png,image/jpeg"
                required
              />
            </div>
            <div class="hint">
              Supported: PDF, PNG, JPG. You can select multiple files at once.
            </div>
            <button type="submit">Ingest documents</button>
          </form>
          <div id="status" class="status"></div>
          <div id="summary" class="summary" style="display:none;"></div>
        </div>
        <script>
          const form = document.getElementById("upload-form");
          const statusEl = document.getElementById("status");
          const summaryEl = document.getElementById("summary");

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            statusEl.textContent = "Uploading...";
            statusEl.className = "status";
            summaryEl.style.display = "none";
            summaryEl.innerHTML = "";

            const formData = new FormData(form);

            try {
              const response = await fetch("/ingest", {
                method: "POST",
                body: formData,
              });

              const isJson = response.headers
                .get("content-type")
                ?.includes("application/json");
              const data = isJson ? await response.json() : null;

              if (response.ok && data?.status === "ok") {
                const files = data.files || [];
                const createdCount = files.filter((f) => f.extraction_created).length;
                const failedCount = files.length - createdCount;
                const okCount = files.filter((f) => f.validation?.status === "ok").length;
                const partialCount = files.filter((f) => f.validation?.status === "partial").length;
                const failedValidationCount = files.filter((f) => f.validation?.status === "failed").length;

                const avgConfValues = files
                  .map((f) => f.quality?.avg_token_confidence)
                  .filter((v) => typeof v === "number");
                const avgConf = avgConfValues.length
                  ? (avgConfValues.reduce((a, b) => a + b, 0) / avgConfValues.length).toFixed(2)
                  : "n/a";

                const allIssues = files.flatMap((f) => f.validation?.issues || []);
                const errorIssues = allIssues.filter((i) => i.level === "error").length;
                const warningIssues = allIssues.filter((i) => i.level === "warning").length;

                statusEl.textContent = "Upload successful.";
                statusEl.className = "status status--success";

                const failedFiles = files
                  .filter((f) => !f.extraction_created)
                  .map(
                    (f) =>
                      `<div class="summary-line">- ${f.original_filename}: extraction failed (${f.extraction_error || "unknown error"})</div>`
                  )
                  .join("");

                summaryEl.innerHTML = `
                  <h2>Extraction Summary</h2>
                  <div class="summary-line">Files uploaded: ${files.length}</div>
                  <div class="summary-line">Extractions created: ${createdCount}</div>
                  <div class="summary-line">Extraction failures: ${failedCount}</div>
                  <div class="summary-line">Validation status - ok: ${okCount}, partial: ${partialCount}, failed: ${failedValidationCount}</div>
                  <div class="summary-line">Average OCR confidence: ${avgConf}</div>
                  <div class="summary-line">Validation issues - errors: ${errorIssues}, warnings: ${warningIssues}</div>
                  ${failedFiles ? `<div class="summary-issues">${failedFiles}</div>` : ""}
                `;
                summaryEl.style.display = "block";
                form.reset();
              } else {
                const detail =
                  (data && (data.detail || data.message)) ||
                  `Status ${response.status}`;
                statusEl.textContent = `Upload failed: ${detail}`;
                statusEl.className = "status status--error";
                summaryEl.style.display = "none";
              }
            } catch (error) {
              statusEl.textContent = "Upload failed: network or server error.";
              statusEl.className = "status status--error";
              summaryEl.style.display = "none";
            }
          });
        </script>
      </body>
    </html>
    """


@app.post("/ingest")
async def ingest_documents(
    files: List[UploadFile] = File(..., description="PDF and image invoice files"),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    ensure_storage_dir()

    ingested_files: List[dict] = []

    for upload in files:
        if upload.content_type not in ACCEPTED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported content type: {upload.content_type}. "
                f"Accepted types: {sorted(ACCEPTED_CONTENT_TYPES)}",
            )

        content = await upload.read()
        if not content:
            raise HTTPException(
                status_code=400, detail=f"File {upload.filename} is empty."
            )

        invoice_id = str(uuid.uuid4())
        invoice_dir = BASE_STORAGE_DIR / invoice_id
        invoice_dir.mkdir(parents=True, exist_ok=True)

        page_image_paths: List[str] = []

        if upload.content_type == "application/pdf":
            # Convert each PDF page to a PNG image.
            pages = convert_from_bytes(content, fmt="png")
            for index, page in enumerate(pages, start=1):
                page = preprocess_image(page)
                image_path = invoice_dir / f"page-{index}.png"
                page.save(image_path, format="PNG")
                page_image_paths.append(str(image_path))
        else:
            # Normalize images to PNG.
            with Image.open(io.BytesIO(content)) as img:
                processed = preprocess_image(img)
                image_path = invoice_dir / "page-1.png"
                processed.save(image_path, format="PNG")
                page_image_paths.append(str(image_path))

        ingested_files.append(
            {
                "invoice_id": invoice_id,
                "original_filename": upload.filename,
                "page_images": page_image_paths,
            }
        )

        # Automatically run extraction right after ingestion so each new
        # invoice folder gets extraction output files without a second API call.
        extraction_error = None
        try:
            extraction, raw = await _run_extraction_pipeline(invoice_id)
            _persist_extraction_files(invoice_id, extraction, raw)
            ingested_files[-1]["extraction_created"] = True
            ingested_files[-1]["quality"] = (
                extraction.quality.model_dump() if extraction.quality else None
            )
            ingested_files[-1]["validation"] = (
                extraction.validation.model_dump() if extraction.validation else None
            )
        except Exception as exc:
            extraction_error = str(exc)
            ingested_files[-1]["extraction_created"] = False
            ingested_files[-1]["extraction_error"] = extraction_error

    return JSONResponse({"status": "ok", "files": ingested_files})


def _extension_from_content_type(content_type: str) -> str:
    if content_type == "application/pdf":
        return ".pdf"
    if content_type in {"image/png"}:
        return ".png"
    if content_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    return ""


@app.get("/ocr/{invoice_id}")
async def run_ocr(invoice_id: str) -> JSONResponse:
    """
    HTTP wrapper that runs OCR on normalized page images and returns
    a layout-preserving representation (tokens grouped into lines with bboxes).
    """
    pages = await _perform_ocr(invoice_id)
    return JSONResponse({"invoice_id": invoice_id, "pages": [p.model_dump() for p in pages]})


async def _perform_ocr(invoice_id: str) -> List[OcrPage]:
    """
    Run OCR on all normalized page images for the given invoice and return
    layout-preserving pages (tokens grouped into lines with bboxes).
    """
    invoice_dir = BASE_STORAGE_DIR / invoice_id
    if not invoice_dir.exists() or not invoice_dir.is_dir():
        raise HTTPException(status_code=404, detail="Invoice not found.")

    page_files = sorted(invoice_dir.glob("page-*.png"))
    if not page_files:
        raise HTTPException(status_code=400, detail="No page images for invoice.")

    pages: List[OcrPage] = []

    for page_index, image_path in enumerate(page_files, start=1):
        with Image.open(image_path) as img:
            width, height = img.size

            data = pytesseract.image_to_data(
                img,
                output_type=Output.DICT,
                config=f"--psm {OCR_PSM}",
            )

            lines_map: Dict[Tuple[int, int], List[OcrToken]] = {}

            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                conf_str = data["conf"][i]
                try:
                    conf = float(conf_str)
                except ValueError:
                    conf = -1.0

                if not text or conf < 0:
                    continue

                left = data["left"][i]
                top = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                # Normalize to 0–1 coordinates.
                x_min = left / width
                y_min = top / height
                x_max = (left + w) / width
                y_max = (top + h) / height

                block_num = data["block_num"][i]
                line_num = data["line_num"][i]
                line_key = (block_num, line_num)
                line_id = f"p{page_index}_b{block_num}_l{line_num}"
                block_id = f"p{page_index}_b{block_num}"

                token = OcrToken(
                    text=text,
                    bbox=(x_min, y_min, x_max, y_max),
                    confidence=conf,
                    line_id=line_id,
                    block_id=block_id,
                )
                lines_map.setdefault(line_key, []).append(token)

            # Build OcrLine objects in reading order: block_num, then line_num, then x.
            ocr_lines: List[OcrLine] = []
            sorted_keys = sorted(lines_map.keys(), key=lambda k: (k[0], k[1]))
            for order_index, key in enumerate(sorted_keys):
                tokens = sorted(
                    lines_map[key], key=lambda t: t.bbox[0]
                )  # sort left-to-right

                # Assign stable token indices within the line.
                tokens = [
                    OcrToken(
                        text=t.text,
                        bbox=t.bbox,
                        confidence=t.confidence,
                        line_id=t.line_id,
                        block_id=t.block_id,
                        token_index=i,
                    )
                    for i, t in enumerate(tokens)
                ]

                # Derive line bbox from token bboxes.
                x_mins = [t.bbox[0] for t in tokens]
                y_mins = [t.bbox[1] for t in tokens]
                x_maxs = [t.bbox[2] for t in tokens]
                y_maxs = [t.bbox[3] for t in tokens]

                line_bbox = (
                    min(x_mins),
                    min(y_mins),
                    max(x_maxs),
                    max(y_maxs),
                )

                line_id = tokens[0].line_id
                ocr_lines.append(
                    OcrLine(
                        id=line_id,
                        tokens=tokens,
                        bbox=line_bbox,
                        reading_order_index=order_index,
                    )
                )

            pages.append(
                OcrPage(
                    page_number=page_index,
                    width=width,
                    height=height,
                    lines=ocr_lines,
                )
            )

    return pages


def _build_llm_extraction_prompt(invoice_id: str, pages: List[OcrPage]) -> str:
    """
    Build a compact, layout-aware prompt from OCR pages to drive the LLM extraction.
    """
    lines_text: List[str] = []
    for page in pages:
        for line in sorted(page.lines, key=lambda l: l.reading_order_index):
            text = " ".join(token.text for token in line.tokens)
            if not text.strip():
                continue
            lines_text.append(f"[page={page.page_number} line_id={line.id}] {text}")

    joined_lines = "\n".join(lines_text[:800])

    schema_description = """
Return a JSON object with the following structure and NO additional commentary:
{
  "line_items": [
    {
      "description": string,
      "quantity": number | null,
      "unit_price": number | null,
      "line_total": number | null,
      "discount": number | null,
      "extra_cost": number | null,
      "source_line_ids": [string, ...]
    }
  ],
  "summary": {
    "subtotal": number | null,
    "total": number | null,
    "currency": string | null,
    "tax_bases": [number, ...] | null,
    "tax_rates": [number, ...] | null
  },
  "metadata": {
    "invoice_number": string | null,
    "invoice_date": string | null,
    "vendor_name": string | null,
    "customer_name": string | null
  }
}
If any value is missing or ambiguous in the text, set it explicitly to null. Do NOT invent values.
"""

    instructions = f"""
You are given OCR-extracted lines from one invoice, with page numbers and stable line_ids.
Your task is to:
- Identify line items (product/service rows) and map them to the fields in the schema.
- Identify summary fields (subtotal, tax bases + rates, total).
- Identify invoice-level metadata (invoice number, date, vendor, customer).
- For each numeric field, only use numbers that appear in the text.
- For each line item, fill source_line_ids with the line_id(s) the row came from.
- For vendor_name and customer_name, copy the exact OCR line text (no abbreviations).
- If something is unclear or not present, use null for that field instead of guessing.

INVOICE_ID={invoice_id}

OCR LINES (in reading order):
{joined_lines}

{schema_description}
"""
    return instructions.strip()


async def _run_extraction_pipeline(invoice_id: str) -> Tuple[InvoiceExtraction, str]:
    """
    End-to-end extraction pipeline:
    OCR -> prompt build -> Gemini call -> JSON parse -> typed mapping.
    Returns (typed_extraction, raw_model_text).
    """
    pages = await _perform_ocr(invoice_id)
    prompt = _build_llm_extraction_prompt(invoice_id, pages)
    model = get_gemini_model()

    try:
        completion = model.generate_content(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM extraction failed: {exc}") from exc

    raw = completion.text or ""

    try:
        parsed = _parse_llm_json(raw)
    except Exception as exc:
        raise RuntimeError(
            f"LLM did not return valid JSON. Raw output starts with: {raw[:200]!r}"
        ) from exc

    extraction = InvoiceExtraction(
        invoice_id=invoice_id,
        line_items=[
            InvoiceLineItem(
                description=item.get("description", "") or "",
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                line_total=item.get("line_total"),
                discount=item.get("discount"),
                extra_cost=item.get("extra_cost"),
                source_line_ids=item.get("source_line_ids") or [],
            )
            for item in parsed.get("line_items", []) or []
        ],
        summary=InvoiceSummary(
            subtotal=parsed.get("summary", {}).get("subtotal"),
            total=parsed.get("summary", {}).get("total"),
            currency=parsed.get("summary", {}).get("currency"),
            tax_bases=parsed.get("summary", {}).get("tax_bases"),
            tax_rates=parsed.get("summary", {}).get("tax_rates"),
        ),
        metadata=InvoiceMetadata(
            invoice_number=parsed.get("metadata", {}).get("invoice_number"),
            invoice_date=parsed.get("metadata", {}).get("invoice_date"),
            vendor_name=parsed.get("metadata", {}).get("vendor_name"),
            customer_name=parsed.get("metadata", {}).get("customer_name"),
        ),
        raw_model_output=parsed,
    )

    # Deterministic metadata correction from OCR anchors to reduce LLM drift.
    ocr_vendor, ocr_customer = _extract_vendor_customer_from_ocr(pages)
    ocr_lines = _collect_ocr_line_texts(pages)
    if ocr_vendor and not _text_in_ocr_lines(extraction.metadata.vendor_name, ocr_lines):
        extraction.metadata.vendor_name = ocr_vendor
    if ocr_customer and not _text_in_ocr_lines(extraction.metadata.customer_name, ocr_lines):
        extraction.metadata.customer_name = ocr_customer

    # Compute and attach OCR quality metrics.
    extraction.quality = _compute_ocr_quality(pages)

    # Attach token-level provenance for numeric fields (line items).
    _attach_token_provenance(extraction, pages)

    # Run zero-hallucination validation on top of the extraction.
    ocr_numbers_by_line = _collect_ocr_numbers_by_line(pages)
    extraction.validation = _validate_extraction(
        invoice_id,
        extraction,
        ocr_numbers_by_line,
        ocr_lines,
    )

    # Add quality-derived warnings to validation output.
    if extraction.validation and extraction.quality:
        if extraction.quality.avg_token_confidence < 70.0:
            extraction.validation.issues.append(
                ValidationIssue(
                    level="warning",
                    code="low_avg_ocr_confidence",
                    message=(
                        f"Average OCR confidence is low ({extraction.quality.avg_token_confidence:.2f}). "
                        "Extraction reliability may be reduced."
                    ),
                    path="quality.avg_token_confidence",
                )
            )
        if extraction.quality.low_confidence_ratio > 0.35:
            extraction.validation.issues.append(
                ValidationIssue(
                    level="warning",
                    code="high_low_confidence_ratio",
                    message=(
                        f"High ratio of low-confidence OCR tokens ({extraction.quality.low_confidence_ratio:.2%}). "
                        "Manual review is recommended."
                    ),
                    path="quality.low_confidence_ratio",
                )
            )
        # Update status if only warnings were added.
        has_error = any(i.level == "error" for i in extraction.validation.issues)
        if not has_error and extraction.validation.issues:
            extraction.validation.status = "partial"

    return extraction, raw


def _persist_extraction_files(invoice_id: str, extraction: InvoiceExtraction, raw: str) -> None:
    """
    Persist parsed extraction and raw model output beside page images.
    """
    invoice_dir = BASE_STORAGE_DIR / invoice_id
    invoice_dir.mkdir(parents=True, exist_ok=True)
    result_path = invoice_dir / "extraction.json"
    raw_path = invoice_dir / "extraction_raw.txt"
    result_path.write_text(
        json.dumps(extraction.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    raw_path.write_text(raw, encoding="utf-8")


def _validate_extraction(
    invoice_id: str,
    extraction: InvoiceExtraction,
    ocr_numbers_by_line: Dict[str, List[str]],
    ocr_lines: List[str],
) -> InvoiceValidation:
    """
    Zero-hallucination validation:
    - Ensure numeric fields appear in OCR text near their source_line_ids.
    - Check line-item arithmetic consistency.
    - Check subtotal/total consistency.
    """
    issues: List[ValidationIssue] = []
    tol = 0.01

    # Line-item checks.
    for idx, item in enumerate(extraction.line_items):
        path_prefix = f"line_items[{idx}]"

        # Provenance checks: quantity, unit_price, line_total, discount, extra_cost.
        for field_name in ("quantity", "unit_price", "line_total", "discount", "extra_cost"):
            value = getattr(item, field_name)
            if value is None:
                continue
            refs = (item.source_tokens or {}).get(field_name) or []
            if not refs:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="missing_token_provenance",
                        message=(
                            f"{field_name}={value!r} could not be linked to OCR tokens "
                            f"for source_line_ids={item.source_line_ids}"
                        ),
                        path=f"{path_prefix}.{field_name}",
                    )
                )

        # Arithmetic check for line_total.
        if item.quantity is not None and item.unit_price is not None and item.line_total is not None:
            discount = item.discount or 0.0
            extra_cost = item.extra_cost or 0.0
            expected = item.quantity * item.unit_price - discount + extra_cost
            if abs(expected - item.line_total) > tol:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="line_total_mismatch",
                        message=(
                            f"Computed line_total {expected:.2f} from quantity, unit_price, "
                            f"discount, extra_cost does not match extracted {item.line_total:.2f}."
                        ),
                        path=f"{path_prefix}.line_total",
                    )
                )

    # Summary subtotal vs sum of line totals.
    if extraction.summary.subtotal is not None:
        sum_lines = sum(
            item.line_total for item in extraction.line_items if item.line_total is not None
        )
        if abs(sum_lines - extraction.summary.subtotal) > tol:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="subtotal_mismatch",
                    message=(
                        f"Sum of line item totals {sum_lines:.2f} does not match extracted "
                        f"subtotal {extraction.summary.subtotal:.2f}."
                    ),
                    path="summary.subtotal",
                )
            )

    # Total vs subtotal + tax.
    if (
        extraction.summary.total is not None
        and extraction.summary.subtotal is not None
        and extraction.summary.tax_bases
        and extraction.summary.tax_rates
        and len(extraction.summary.tax_bases) == len(extraction.summary.tax_rates)
    ):
        computed_tax = 0.0
        for base, rate in zip(extraction.summary.tax_bases, extraction.summary.tax_rates):
            if base is None or rate is None:
                continue
            computed_tax += float(base) * float(rate) / 100.0

        expected_total = extraction.summary.subtotal + computed_tax
        if abs(expected_total - extraction.summary.total) > tol:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="total_mismatch",
                    message=(
                        f"Subtotal + computed tax ({expected_total:.2f}) does not match extracted "
                        f"total {extraction.summary.total:.2f}."
                    ),
                    path="summary.total",
                )
            )

    # Metadata text presence checks.
    if extraction.metadata.vendor_name and not _text_in_ocr_lines(
        extraction.metadata.vendor_name, ocr_lines
    ):
        issues.append(
            ValidationIssue(
                level="error",
                code="vendor_not_in_ocr",
                message=(
                    f"Vendor name {extraction.metadata.vendor_name!r} not found in OCR text."
                ),
                path="metadata.vendor_name",
            )
        )

    if extraction.metadata.customer_name and not _text_in_ocr_lines(
        extraction.metadata.customer_name, ocr_lines
    ):
        issues.append(
            ValidationIssue(
                level="warning",
                code="customer_not_in_ocr",
                message=(
                    f"Customer name {extraction.metadata.customer_name!r} not found in OCR text."
                ),
                path="metadata.customer_name",
            )
        )

    # Determine overall status.
    has_error = any(i.level == "error" for i in issues)
    status: Literal["ok", "partial", "failed"]
    if has_error:
        status = "failed"
    elif issues:
        status = "partial"
    else:
        status = "ok"

    return InvoiceValidation(status=status, issues=issues)


@app.get("/extract/{invoice_id}")
async def extract_invoice(invoice_id: str) -> JSONResponse:
    """
    Run OCR for the invoice, then use an LLM to propose a structured extraction
    following the InvoiceExtraction schema. This is an initial extraction only;
    further zero-hallucination validation happens in downstream steps.
    """
    try:
        extraction, raw = await _run_extraction_pipeline(invoice_id)
        _persist_extraction_files(invoice_id, extraction, raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(extraction.model_dump())


@app.get("/analysis/{invoice_id}")
async def get_analysis(invoice_id: str) -> JSONResponse:
    """
    Return a compact analysis view focused on validation + OCR quality.
    Uses extraction.json if available, otherwise runs extraction pipeline once.
    """
    result_path = BASE_STORAGE_DIR / invoice_id / "extraction.json"
    payload: dict

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        extraction, raw = await _run_extraction_pipeline(invoice_id)
        _persist_extraction_files(invoice_id, extraction, raw)
        payload = extraction.model_dump()

    return JSONResponse(
        {
            "invoice_id": invoice_id,
            "quality": payload.get("quality"),
            "validation": payload.get("validation"),
        }
    )

