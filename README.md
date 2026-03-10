# Zero-Error-Invoice-Data-Extraction

## Automated validation tests

This repository includes a small fixture-based test suite for the validation engine.

### Covered sample set

- clean single-page invoice
- multi-page invoice
- noisy scan invoice
- tricky inconsistent totals invoice (deliberate mismatch)

### Run tests

```bash
pip install -r requirements.txt
pytest -q
```
