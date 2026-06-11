# Anesthesia Record Extraction

Tools for extracting structured monitoring data from scanned anesthesia record PDFs or images.

The pipeline was developed for fixed-layout anesthesia records with a colored monitoring chart. It estimates:

- Systolic/diastolic blood pressure from red arterial-pressure traces
- Heart rate from green traces
- SpO2 and EtCO2 values from the monitoring table
- Alignment/debug images for manual review
- CSV/XLSX review packages

No patient source records or generated review outputs are included in this repository.

## Directory Layout

Place local case data beside the scripts. The default scanner expects grouped case folders:

```text
anesthesia_record_extraction/
  正常/
    case_001/
      record.pdf
      optional_case_export.json
  阳性/
    case_002/
      record.jpg
```

Each case folder may contain a PDF, JPG, JPEG, or PNG. If a Cadwell-style JSON export is present, the batch script can use it as a fallback for case start time when anesthesia-time OCR fails.

Generated outputs are ignored by Git:

- `批量监测CSV/`
- `批量气体识别CSV/`
- `批量坐标对齐图/`
- `人工核对包/`

## Install

```powershell
cd anesthesia_record_extraction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`rapidocr-onnxruntime` is optional but recommended. Without it, time recognition falls back to template-based digit matching and may require more manual review.

## Run

Extract blood pressure, heart rate, and a first-pass SpO2/EtCO2 lookup:

```powershell
python .\batch_extract_monitoring_cases.py --workspace .
```

Run the dedicated SpO2/EtCO2 extractor:

```powershell
python .\batch_extract_spo2_etco2.py --workspace .
```

Build a manual review package after batch extraction:

```powershell
python .\build_manual_review_package.py
python .\build_missing_trace_review.py
```

If your case groups are not named `正常` and `阳性`, pass them explicitly:

```powershell
python .\batch_extract_monitoring_cases.py --workspace D:\cases --groups normal positive
python .\batch_extract_spo2_etco2.py --workspace D:\cases --groups normal positive
python .\build_manual_review_package.py --workspace D:\cases
python .\build_missing_trace_review.py --workspace D:\cases
```

## Notes

- The extraction logic assumes a fixed chart layout. Adjust ROI constants in `batch_extract_monitoring_cases.py` for other form templates.
- The public version intentionally does not ship private reference PDFs or labeled digit templates.
- `extract_high_confidence_monitor.py` is a single-case experimental helper. Fill local `PAGE_CONFIGS` before running it.
- Outputs are designed for human review; high-confidence values should still be checked against the source record before clinical or research use.
