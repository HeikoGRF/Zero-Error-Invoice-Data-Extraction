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
      - `page_images`: list of normalized image paths (one per page).

### Storage Layout

Every uploaded document is normalized to **PNG images**:

- For PDFs: each page is rendered as an image and stored as:
  - `storage/ingested/<invoice_id>/page-1.png`
  - `storage/ingested/<invoice_id>/page-2.png`
  - ...
- For images: the original is re-encoded to PNG and stored as:
  - `storage/ingested/<invoice_id>/page-1.png`

Later stages (OCR, layout analysis) will use `invoice_id` and `page_images` to load the normalized page images and generate OCR outputs.

