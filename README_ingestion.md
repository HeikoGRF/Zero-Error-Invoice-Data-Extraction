## Document Ingestion Service

This is the first step of the Zero-Error Invoice Data Extraction pipeline. It is responsible for accepting invoice documents, validating them, and storing them in a normalized directory structure for downstream OCR and extraction.

### Stack

- **Backend**: FastAPI
- **Supported formats**: PDFs and images (`.png`, `.jpg`, `.jpeg`)

### API

- **POST** `/ingest`
  - **Body**: `multipart/form-data` with one or more files under the `files` field.
  - **Accepted content types**:
    - `application/pdf`
    - `image/png`
    - `image/jpeg`
    - `image/jpg`
  - **Response**:
    - `status`: `"ok"` when ingestion succeeds.
    - `files`: list of ingested file descriptors:
      - `invoice_id`: UUID assigned per uploaded file (used as directory name).
      - `original_filename`
      - `stored_path`: relative path on disk for downstream stages.
      - `content_type`

### Storage Layout

Files are stored under `storage/ingested/<invoice_id>/page-1.<ext>`.

Later stages (OCR, layout analysis) will use the `invoice_id` and stored path to load the original binary content and generate OCR outputs.

