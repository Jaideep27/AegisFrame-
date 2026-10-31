# Phase 1b spike: OCR and layout (2026-10-30)

## What is installed
- **RapidOCR 3.10.0** (Apache-2.0, ONNX, CPU) defaults to **PP-OCRv6** (`multi_PP-OCRv6_det_small` /
  `rec_small`), which is newer than the plan assumed. Its one multilingual model covers Chinese,
  Japanese, English and the Latin-script languages (es, pt, fr, de, id, sw, tl, vi…).
- **Arabic and Devanagari are not in PP-OCRv6.** RapidOCR still ships PP-OCRv5 recognisers for them
  (`arabic_`, `devanagari_`, `cyrillic_`, `korean_`, `th_`, `ta_`, `te_` …). The runtime must choose the
  recogniser by script.
- **RapidTable 3.0.2** (Apache-2.0) is on PyPI, ONNX, and needs no torch. It is not installed yet; see
  "Next" below.

## Layout rebuild (`aegisframe.ocr.layout.from_lines`)
This replaces `ground.pair_fields`. It works from geometry only:
1. undo the page tilt, using the median baseline angle that RapidOCR's quadrilateral boxes give
   (`OcrLine.angle`, new);
2. group lines into rows;
3. a text-only row of 3+ cells starts a table, with the header fixing the column positions;
4. 2-cell rows and "label: value" cells become fields;
5. the tallest early line is the title.

## Results (`training/spike/ocr_spike.py --n 24`)
The test images are 24 records rendered from `doclib`: 8 families × 8 locales. Each record is tested
clean, as a scan (slight blur, JPEG) and as a phone photo (±3° rotation, uneven light, JPEG 35–60).
Machine: the user's laptop, CPU throttled to about 1.4 GHz.

| Condition | CER | Field recall | Table-cell accuracy | Latency p50 |
|---|---|---|---|---|
| clean | 0.076 | 0.974 | 0.938 | 2.3 s |
| scan | 0.062 | 0.964 | 0.920 | 2.3 s |
| photo | 0.066 | 0.948 | 0.960 | 2.3 s |

These are our own rendered records, not real scans, so they show whether the layout logic works. They
are not a real-world accuracy figure.

**Before the tilt correction**, phone photos scored CER 0.257 and table-cell accuracy 0.297. Rows slanted
across the page and labels split from their values. **After it**, photos score as well as clean images.

Per family (clean), field recall / cell accuracy:

| Family | Field recall | Cell accuracy |
|---|---|---|
| marksheet | 1.00 | 0.98 |
| payslip | 0.96 | 1.00 |
| bank statement | 1.00 | 0.98 |
| tax invoice | 1.00 | 0.88 |
| lab report | 0.83 | 0.85 |

## Remaining errors are recognition, not layout
On a rendered phone photo of a marksheet, the structure is now right: the subject / grade / points rows
come back intact. The recogniser still misreads digits:
- "7.33" read as "733" (the decimal point is lost);
- "08" read as "80";
- "05" read as "00".

The ₹ sign is also sometimes dropped. These are exactly the confusions `datagen_v2/ocrsim.py`
injects into training contexts, so the model learns not to trust a single digit.

## Next
- Run the user's real marksheet photo: `--image <path>`. The bar is correct rows and GPA 9.7.
- Try `rec_medium` (PP-OCRv6) and the PP-OCRv5 server recogniser against the digit errors, and measure
  their latency cost.
- Script routing: when Arabic or Devanagari is detected, re-recognise with the PP-OCRv5 model for that
  script.
- RapidTable for ruled or dense tables, compared with the geometric rebuild on the same images.
- Latency: about 2.3 s per image on this throttled CPU. Check the `tiny` det/rec models and a downscaled
  input.
