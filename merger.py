import os
import re

INPUT_FOLDER = "ocr_results"  # adjust as needed
OUTPUT_FILE = "merged_document.txt"

# Regex to extract page number


def extract_page_number(filename):
    match = re.search(r'page_(\d+)\.txt', filename)
    return int(match.group(1)) if match else -1


def merge_text_files(input_folder, output_file):
    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".txt") and re.match(r'page_\d+\.txt', f)
    ]
    files_sorted = sorted(files, key=extract_page_number)

    with open(output_file, "w", encoding="utf-8") as out_file:
        for fname in files_sorted:
            path = os.path.join(input_folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                # out_file.write(f"--- {fname} ---\n")
                out_file.write(f.read().strip() + "\n\n")

    print(f"✅ Merged {len(files_sorted)} files into '{output_file}'")


if __name__ == "__main__":
    merge_text_files(INPUT_FOLDER, OUTPUT_FILE)
