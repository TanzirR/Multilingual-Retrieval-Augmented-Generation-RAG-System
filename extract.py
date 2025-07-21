from pdf2image import convert_from_path
import pytesseract
import os

# Set up Tesseract for Bangla
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update if needed

# Optional: Add Bangla language if not already installed
# Language code for Bengali is "ben"

def ocr_bangla_pdf(pdf_path, lang='ben'):
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang=lang)
        full_text += text.strip() + "\n\n"
        print(f"[INFO] Processed page {i+1}")
    
    return full_text

# Run
pdf_path = "bangla-text.pdf"
extracted_text = ocr_bangla_pdf(pdf_path)

with open("bangla_ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("✅ OCR extraction complete.")
