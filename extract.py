import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import unicodedata
import re

def clean_ocr_text(text):
    """
    Cleans OCR output by removing extra spaces, OCR noise, and known artifacts.
    """
    text = re.sub(r'\f', ' ', text)                       # Remove form feeds
    text = re.sub(r'\n+', '\n', text)                     # Collapse newlines
    text = re.sub(r'[ ]+', ' ', text)                     # Collapse spaces
    text = re.sub(r'\d+[\]\'\d+]', '', text)              # Remove junk like '169]0'
    text = re.sub(r'\b[i]{2,}\)', '', text)               # Remove Roman numerals like 'iii)'
    return text.strip()

def extract_text_with_ocr(pdf_path, dpi=400, lang='ben', debug=False):
    """
    Extracts OCR text from image-based Bangla PDFs with optional debugging.
    """
    print(f"📄 Converting PDF to images at {dpi} DPI...")
    images = convert_from_path(pdf_path, dpi=dpi)

    full_text = ""
    for i, img in enumerate(images):
        print(f"🔍 OCR processing page {i+1}/{len(images)}...")

        # Use Bengali Tesseract model with better config
        custom_config = r'--oem 1 --psm 6'  # You may also try psm=3 or 11 based on layout
        raw_text = pytesseract.image_to_string(img, lang=lang, config=custom_config)

        # Unicode normalization
        normalized_text = unicodedata.normalize('NFC', raw_text)

        # Clean noisy OCR output
        cleaned_text = clean_ocr_text(normalized_text)

        # Optionally log page number
        if debug:
            full_text += f"\n\n--- PAGE {i+1} ---\n{cleaned_text}\n"
        else:
            full_text += cleaned_text + "\n"

    return full_text.strip()


# --- Execution ---
pdf_file_path = 'bangla-text.pdf'
output_file = 'extracted_output.txt'

print("🔄 Starting OCR text extraction...")
extracted_text = extract_text_with_ocr(pdf_file_path, dpi=300, lang='ben', debug=True)

# Save to file
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"✅ Extracted text saved to: {output_file}")
    print(f"📊 Total characters: {len(extracted_text)}")
    print(f"📄 Total lines: {len(extracted_text.splitlines())}")
except Exception as e:
    print(f"❌ Error saving file: {e}")

print("✅ OCR extraction and saving complete!")