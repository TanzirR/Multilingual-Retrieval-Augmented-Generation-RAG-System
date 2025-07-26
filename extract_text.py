from pdf2image import convert_from_path
from PIL import Image
import unicodedata
import re
import json 
from collections import defaultdict
import easyocr  

class DocumentProcessor:
    def __init__(self, pages_data):
        self.pages_data = pages_data
        print(f" Processing {len(self.pages_data)} pages.")

    def clean_text_for_detection(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def segment_document_by_type(self):
        print("\n Segmenting document into story, vocabulary, and question blocks...")
        segments = []
        current_type = None
        current_content_pages = []
        current_start_page = -1
        type_counts = defaultdict(int)

        QUESTION_PATTERNS = r'পাঠ্যপুস্তকের\s*প্রশ্ন|বহুনির্বাচনী|উদ্দীপক|প্রশ্নের\s*উত্তর'
        STORY_PATTERNS = r'মূল\s*গল্প|অনুপমের\s*বযস\s*সাতাশ'
        VOCABULARY_PATTERNS = r'শব্দার্থ\s*ও\s*টীকা|মূল\s*শব্দ'

        for page_data in self.pages_data:
            page_num = page_data['page_num']
            cleaned_page_text = self.clean_text_for_detection(page_data['text'])

            detected_type = "general"
            if re.search(QUESTION_PATTERNS, cleaned_page_text):
                detected_type = "questions"
            elif re.search(STORY_PATTERNS, cleaned_page_text):
                detected_type = "story"
            elif re.search(VOCABULARY_PATTERNS, cleaned_page_text):
                detected_type = "vocabulary"

            if current_type is None:
                current_type = detected_type
                current_start_page = page_num
                current_content_pages.append(page_data['text'])
            elif detected_type != current_type:
                type_counts[current_type] += 1
                segment_id = f"{current_type}_{type_counts[current_type]}"
                segments.append({
                    "type": current_type,
                    "id": segment_id,
                    "pages": f"{current_start_page}-{page_num - 1}",
                    "text": "\n".join(current_content_pages),
                    "content_preview": "\n".join(current_content_pages)[:100].replace('\n', ' ') + '...'
                })
                current_type = detected_type
                current_start_page = page_num
                current_content_pages = [page_data['text']]
            else:
                current_content_pages.append(page_data['text'])

        if current_type is not None:
            type_counts[current_type] += 1
            segment_id = f"{current_type}_{type_counts[current_type]}"
            segments.append({
                "type": current_type,
                "id": segment_id,
                "pages": f"{current_start_page}-{self.pages_data[-1]['page_num']}",
                "text": "\n".join(current_content_pages),
                "content_preview": "\n".join(current_content_pages)[:100].replace('\n', ' ') + '...'
            })

        print(f"Document segmented into {len(segments)} logical blocks.")
        for segment in segments:
            print(f"   - Type: {segment['type']}, ID: {segment['id']}, Pages: {segment['pages']} | Content Preview: '{segment['content_preview']}'")

        return segments

def clean_ocr_text(text):
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ ]+', ' ', text)
    text = re.sub(r'[\f]+', ' ', text)
    text = re.sub(r'[ ]+', ' ', text)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    text = re.sub(r'\b[i]{2,}\)', '', text)
    return text.strip()

def extract_text_with_ocr(pdf_path, dpi=400, lang_list=['bn', 'en']):
    print(f"Converting PDF to images at {dpi} DPI...")
    images = convert_from_path(pdf_path, dpi=dpi)

    # ✅ Initialize EasyOCR reader once
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(lang_list, gpu=True)

    extracted_pages_data = []
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"OCR processing page {page_num}/{len(images)}...")

        # ✅ Convert PIL Image to format compatible with EasyOCR
        img_np = np.array(img.convert('RGB'))

        results = reader.readtext(img_np, detail=0, paragraph=True)
        raw_text = "\n".join(results)

        normalized_text = unicodedata.normalize('NFC', raw_text)
        cleaned_text = clean_ocr_text(normalized_text)

        extracted_pages_data.append({
            "page_num": page_num,
            "lang": ",".join(lang_list),
            "text": cleaned_text
        })

    return extracted_pages_data

# --- Execution ---
import numpy as np  # ✅ Needed for np.array conversion

pdf_file_path = './data/statement-of-interest.pdf'
output_json_file = 'extracted_pages_data.json'
segmented_output_json_file = 'segmented_document_data.json'

print("Starting OCR text extraction and structuring...")
extracted_data = extract_text_with_ocr(pdf_file_path, dpi=300, lang_list=['bn', 'en'])

try:
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    print(f"Extracted structured page data saved to: {output_json_file}")
    print(f"Total pages extracted: {len(extracted_data)}")
except Exception as e:
    print(f"Error saving extracted_pages_data.json: {e}")

print("\n--- Starting Document Segmentation ---")
processor = DocumentProcessor(extracted_data)
segmented_data = processor.segment_document_by_type()

try:
    with open(segmented_output_json_file, 'w', encoding='utf-8') as f:
        json.dump(segmented_data, f, ensure_ascii=False, indent=2)
    print(f"Segmented document data saved to: {segmented_output_json_file}")
except Exception as e:
    print(f"Error saving segmented_document_data.json: {e}")

print("\nOCR extraction, structuring, and segmentation complete!")
