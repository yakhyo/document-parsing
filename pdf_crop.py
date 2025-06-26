import os
import fitz  # type: ignore
import argparse
from tqdm import tqdm

# --- Static Crop Setting ---
CROP_RECT = (10, 20, 320, 520)  # Set to None for full-page
DPI = 300
# ----------------------------


def extract_images(pdf_path, output_folder, start_page, end_page):
    os.makedirs(output_folder, exist_ok=True)
    document = fitz.open(pdf_path)
    total_pages = len(document)

    end_page = end_page if end_page is not None else total_pages
    page_range = range(start_page, min(end_page, total_pages))

    for i in tqdm(page_range, desc="Processing pages", unit="page"):
        page = document[i]

        try:
            if CROP_RECT:
                rect = fitz.Rect(*CROP_RECT)
                pix = page.get_pixmap(clip=rect, dpi=DPI)
            else:
                pix = page.get_pixmap(dpi=DPI)

            output_path = os.path.join(output_folder, f"page_{i + 1}.png")
            pix.save(output_path)
        except Exception as e:
            tqdm.write(f"Error on page {i + 1}: {e}")

    document.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF to cropped images")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start page (0-based index)")
    parser.add_argument("--end", type=int, default=None, help="End page (exclusive)")

    args = parser.parse_args()
    
    if os.path.exists(args.pdf) and os.path.isfile(args.pdf):
        extract_images(args.pdf, args.out, args.start, args.end)
    else:
        raise FileNotFoundError(f"File not found: {args.pdf}")
