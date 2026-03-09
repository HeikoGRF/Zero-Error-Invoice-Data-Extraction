from pathlib import Path
import io
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pdf2image import convert_from_bytes
from PIL import Image

ACCEPTED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
}

BASE_STORAGE_DIR = Path("storage") / "ingested"


def ensure_storage_dir() -> None:
    BASE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Zero-Error Invoice Extraction - Ingestion")


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
          }
          .status--success {
            color: #4ade80;
          }
          .status--error {
            color: #f97373;
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
        </div>
        <script>
          const form = document.getElementById("upload-form");
          const statusEl = document.getElementById("status");

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            statusEl.textContent = "Uploading...";
            statusEl.className = "status";

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
                statusEl.textContent = "Upload successful.";
                statusEl.className = "status status--success";
                form.reset();
              } else {
                const detail =
                  (data && (data.detail || data.message)) ||
                  `Status ${response.status}`;
                statusEl.textContent = `Upload failed: ${detail}`;
                statusEl.className = "status status--error";
              }
            } catch (error) {
              statusEl.textContent = "Upload failed: network or server error.";
              statusEl.className = "status status--error";
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
                image_path = invoice_dir / f"page-{index}.png"
                page.save(image_path, format="PNG")
                page_image_paths.append(str(image_path))
        else:
            # Normalize images to PNG.
            with Image.open(io.BytesIO(content)) as img:
                rgb_img = img.convert("RGB")
                image_path = invoice_dir / "page-1.png"
                rgb_img.save(image_path, format="PNG")
                page_image_paths.append(str(image_path))

        ingested_files.append(
            {
                "invoice_id": invoice_id,
                "original_filename": upload.filename,
                "page_images": page_image_paths,
            }
        )

    return JSONResponse({"status": "ok", "files": ingested_files})


def _extension_from_content_type(content_type: str) -> str:
    if content_type == "application/pdf":
        return ".pdf"
    if content_type in {"image/png"}:
        return ".png"
    if content_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    return ""

