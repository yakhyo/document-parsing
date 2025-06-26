import os
import argparse
from PIL import Image # type: ignore
import torch # type: ignore
from tqdm import tqdm
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig # type: ignore
from qwen_vl_utils import process_vision_info # type: ignore

# --- Static Config ---
MODEL_ID = "prithivMLmods/docscopeOCR-7B-050425-exp"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4096

# --- BitsAndBytes 4-bit Quantization ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# --- Load Model & Processor ---
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# --- Inference Function ---
def infer(image: Image.Image, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=False,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text

# --- Main Function ---
def main(input_folder, output_folder, prompt):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )

    for img_file in tqdm(image_files, desc="OCR Processing", unit="image"):
        img_path = os.path.join(input_folder, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            ocr_text = infer(image, prompt)
            txt_filename = os.path.splitext(img_file)[0] + ".txt"
            with open(os.path.join(output_folder, txt_filename), "w", encoding="utf-8") as f:
                f.write(ocr_text)
        except Exception as e:
            tqdm.write(f"Error processing {img_file}: {e}")

    print("✅ OCR completed for all images.")

# --- CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on image files using Qwen-VL.")
    parser.add_argument("--input", default="cropped_images", help="Folder with input images")
    parser.add_argument("--output", default="ocr_results", help="Folder to save text outputs")
    parser.add_argument("--prompt", default="Extract all text from this document.", help="Prompt for the model")

    args = parser.parse_args()
    main(args.input, args.output, args.prompt)
