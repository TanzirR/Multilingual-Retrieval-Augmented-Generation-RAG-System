import pytesseract
from pdf2image import convert_from_path
from PIL import ImageOps, ImageFilter # These are imported but not used, can remove if not planning image pre-processing
import langdetect
import os
import unicodedata
import re

# --- 1. Language Detection Function (MUST be defined before ocr_pdf_dynamic) ---
def detect_language(image, sample_lines=5):
    try:
        # Extract a light preview text
        text = pytesseract.image_to_string(image, lang='ben+eng', config="--psm 6")
        lines = text.split('\n')
        sample_text = ' '.join([line.strip() for line in lines if line.strip()][:sample_lines])
        detected_lang = langdetect.detect(sample_text)
    except langdetect.lang_detect_exception.LangDetectException:
        return 'ben+eng'  # Fallback if nothing useful is detected

    if detected_lang.startswith('bn'):
        return 'ben'
    elif detected_lang.startswith('en'):
        return 'eng'
    else:
        return 'ben+eng'

# --- 2. New Post-processing Function (Can be defined before or after ocr_pdf_dynamic, but before its call) ---
def convert_bengali_words_to_digits(text):
    """
    Converts specific Bengali number words to their corresponding Bengali digits
    within the given text, primarily for MCQ options.
    """
    conversion_map = {
        'এক': '১', 'দুই': '২', 'তিন': '৩', 'চার': '৪', 'পাঁচ': '৫',
        'ছয়': '৬', 'সাত': '৭', 'আট': '৮', 'নয়': '৯', 'দশ': '১০',
        'এগারো': '১১', 'বারো': '১২', 'তেরো': '১৩', 'চৌদ্দ': '১৪', 'পনেরো': '১৫',
        'ষোলো': '১৬', 'সতেরো': '১৭', 'আঠারো': '১৮', 'উনিশ': '১৯', 'বিশ': '২০',
        'একুশ': '২১', 'বাইশ': '২২', 'তেইশ': '২৩', 'চব্বিশ': '২৪', 'পঁচিশ': '২৫',
        'ছাব্বিশ': '২৬', 'সাতাশ': '২৭', 'আটাশ': '২৮', 'ঊনত্রিশ': '২৯', 'ত্রিশ': '৩০',
        # Add more if necessary, but focus on the numbers you expect in options
    }

    # This regex looks for patterns like: (ক) পঁচিশ, (খ) ছাব্বিশ, etc.
    # It tries to be specific to avoid unintended replacements elsewhere.
    # It captures the prefix (ক) or (খ), the number word, and the suffix " বছর" or " দিন"
    
    # Pattern 1: (ক) [number_word] বছর
    # Use re.escape() for dictionary keys to handle special regex characters if any
    pattern_bochor = r'(\([কখগঘ]\)\s*)(' + '|'.join(re.escape(k) for k in conversion_map.keys()) + r')(\s*বছর)'
    
    def replace_bochor_match(match):
        prefix = match.group(1)
        number_word = match.group(2)
        suffix = match.group(3)
        return prefix + conversion_map.get(number_word, number_word) + suffix

    text = re.sub(pattern_bochor, replace_bochor_match, text)

    # You might want a similar pattern for "দিন" if it ever appears as words:
    # pattern_din = r'(\([কখগঘ]\)\s*)(' + '|'.join(re.escape(k) for k in conversion_map.keys()) + r')(\s*দিন)'
    # text = re.sub(pattern_din, replace_din_match, text) # You'd need a similar replace function

    return text

# --- 3. Main OCR Function ---
def ocr_pdf_dynamic(pdf_path):
    images = convert_from_path(pdf_path)
    full_text = ""

    bengali_vowels = "অআইঈউঊঋএঐওঔ"
    bengali_consonants = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
    bengali_vowel_signs = "ািীুূৃেৈোৌ্র্য"
    bengali_diacritics = "ঁংঃ"
    bengali_digits = "০১২৩৪৫৬৭৮৯"
    
    # FIX: Escaped the double quote character
    bengali_punctuation = ".,!?;:'\\\"-—()[]{}/\\|&@#$%*+=<> " 
    
    whitelist_chars = (
        bengali_vowels + bengali_consonants + bengali_vowel_signs +
        bengali_diacritics + bengali_digits + bengali_punctuation
    )
    whitelist_chars = unicodedata.normalize('NFC', whitelist_chars)

    if '"' in whitelist_chars and '\\"' not in whitelist_chars:
        print("WARNING: Unescaped double quote found in whitelist_chars. This might cause issues.")

    final_custom_config = f"--oem 1 --psm 3 -c tessedit_char_whitelist=\"{whitelist_chars}\""

    for i, img in enumerate(images):
        lang = detect_language(img) # detect_language is now defined above
        print(f"[INFO] Page {i+1}: Detected language = {lang}")

        raw_text = pytesseract.image_to_string(img, lang=lang, config=final_custom_config)
        full_text += f"\n\n--- Page {i+1} [Lang: {lang}] ---\n{raw_text.strip()}\n"

    processed_text = convert_bengali_words_to_digits(full_text)
    return processed_text

# --- 4. Execution Block ---
if __name__ == "__main__":
    pdf_path = "bangla-text.pdf"
    output_path = "ocr_auto_dynamic_output.txt"

    print(f"[INFO] Starting OCR with per-page language detection and post-processing...")
    extracted_text = ocr_pdf_dynamic(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"✅ OCR extraction complete. Output saved to '{output_path}'")