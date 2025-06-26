# PDF OCR Pipeline

A modular pipeline to crop pages from a PDF, perform OCR on cropped images using a vision-language model, and merge the output into a single document.

---

## Folder Structure

```
.
├── input.pdf               # Input PDF document
├── pdf_crop.py             # Crops PDF pages into images
├── doc_parse.py            # OCR using Qwen2.5-VL
├── merger.py               # Merges per-page text outputs
├── merged_document.txt     # Final merged result
├── requirements.txt        # Dependency list
└── README.txt              # Documentation (you’re reading this)
```

---

## Setup

Install dependencies:
`pip install -r requirements.txt`
Make sure your environment supports GPU and you have access to:
`prithivMLmods/docscopeOCR-7B-050425-exp`

---

## Workflow

### 1. Crop PDF Pages

    ```
    python pdf_crop.py --pdf input.pdf --out cropped_images --start 199 --end 209
    ```

- Crops selected pages from `input.pdf`
- Saves cropped images to `cropped_images/`
- Uses static `CROP_RECT` configured in script

---

### 2. Run OCR on Cropped Images

    ```
    python doc_parse.py --input cropped_images --output ocr_results
    ```

- Uses Qwen2.5-VL model with 4-bit quantization (still slow, almost the same with not quantized version)
- Saves `.txt` files per image to `ocr_results/`

---

### 3. Merge OCR Outputs

    python merger.py

- Merges all `ocr_results/page_*.txt` files in page order
- Saves combined result to `merged_document.txt`

---

## Notes

- Page files must be named `page_<number>.txt`
- You can modify `CROP_RECT` for different page regions
- GPU recommended for fast inference

---

## To Do

- Inference Optimization
- Add smart cropping
- Enable multiprocessing for OCR
