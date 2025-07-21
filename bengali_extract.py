from pdf2image import convert_from_path
from PIL import ImageOps, ImageFilter 
import pytesseract
import os

def ocr_bangla_pdf(pdf_path, lang='ben'):
    custom_config = "--oem 1 --psm 3"
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang=lang, config=custom_config)
        full_text += text.strip() + "\n\n"
        print(f"[INFO] Processed page {i+1}")
    return full_text

# Run
pdf_path = "bangla-text.pdf"
extracted_text = ocr_bangla_pdf(pdf_path)

with open("bangla_ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("OCR extraction complete.")
