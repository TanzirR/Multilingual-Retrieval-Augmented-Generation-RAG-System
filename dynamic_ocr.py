import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import unicodedata
import re
import json 
from collections import defaultdict

class DocumentProcessor:
    def __init__(self, pages_data):
        """
        Initializes the DocumentProcessor with raw page data.
        pages_data: A list of dictionaries, where each dict contains 'page_num', 'lang', and 'text'.
        """
        self.pages_data = pages_data
        print(f" Processing {len(self.pages_data)} pages.")

    def clean_text_for_detection(self, text):
        """
        Cleans text for pattern detection by normalizing whitespace.
        """
        # Replace multiple whitespace characters with a single space and strip leading/trailing whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def segment_document_by_type(self):
        """
        Segments the document into logical blocks (story, vocabulary, questions, general)
        based on content patterns and page transitions.
        """
        print("\n Segmenting document into story, vocabulary, and question blocks...")
        segments = []
        current_type = None
        current_content_pages = [] # Store list of page texts for current segment
        current_start_page = -1
        type_counts = defaultdict(int)

        # Define patterns for page type detection
        QUESTION_PATTERNS = r'পাঠ্যপুস্তকের\s*প্রশ্ন|বহুনির্বাচনী|উদ্দীপক|প্রশ্নের\s*উত্তর'
        STORY_PATTERNS = r'মূল\s*গল্প|অনুপমের\s*বযস\s*সাতাশ'
        VOCABULARY_PATTERNS = r'শব্দার্থ\s*ও\s*টীকা|মূল\s*শব্দ'

        for page_data in self.pages_data:
            page_num = page_data['page_num']
            cleaned_page_text = self.clean_text_for_detection(page_data['text'])

            # Determine page type with explicit precedence (Questions > Story > Vocabulary > General)
            detected_type = "general"
            if re.search(QUESTION_PATTERNS, cleaned_page_text):
                detected_type = "questions"
            elif re.search(STORY_PATTERNS, cleaned_page_text):
                detected_type = "story"
            elif re.search(VOCABULARY_PATTERNS, cleaned_page_text):
                detected_type = "vocabulary"

            if current_type is None:
                # Initialize the very first segment
                current_type = detected_type
                current_start_page = page_num
                current_content_pages.append(page_data['text'])
            elif detected_type != current_type:
                # Type has changed, finalize the previous segment
                type_counts[current_type] += 1
                segment_id = f"{current_type}_{type_counts[current_type]}"
                segments.append({
                    "type": current_type,
                    "id": segment_id,
                    "pages": f"{current_start_page}-{page_num - 1}",
                    "text": "\n".join(current_content_pages),
                    "content_preview": "\n".join(current_content_pages)[:100].replace('\n', ' ') + '...'
                })

                # Start a new segment
                current_type = detected_type
                current_start_page = page_num
                current_content_pages = [page_data['text']]
            else:
                # Same type, continue the current segment
                current_content_pages.append(page_data['text'])

        # Finalize the last segment after the loop
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
# --- End of DocumentProcessor ---


def clean_ocr_text(text):
    """
    Cleans OCR output by removing extra spaces, OCR noise, and known artifacts.
    """
    text = re.sub(r'\f', ' ', text)                  # Remove form feeds
    text = re.sub(r'\n+', '\n', text)                 # Collapse newlines
    text = re.sub(r'[ ]+', ' ', text)                 # Collapse spaces
    text = re.sub(r'\d+[\]\'\d+]', '', text)          # Remove junk like '169]0'
    text = re.sub(r'\b[i]{2,}\)', '', text)           # Remove Roman numerals like 'iii)'
    return text.strip()

def extract_text_with_ocr(pdf_path, dpi=400, lang='ben'):
    """
    Extracts OCR text from image-based Bangla PDFs and returns structured page data.
    """
    print(f"Converting PDF to images at {dpi} DPI...")
    images = convert_from_path(pdf_path, dpi=dpi)

    extracted_pages_data = [] # Changed to store structured data
    for i, img in enumerate(images):
        page_num = i + 1
        print(f"OCR processing page {page_num}/{len(images)}...")

        # Use Bengali Tesseract model with better config
        custom_config = r'--oem 1 --psm 6'
        raw_text = pytesseract.image_to_string(img, lang=lang, config=custom_config)

        # Unicode normalization
        normalized_text = unicodedata.normalize('NFC', raw_text)

        # Clean noisy OCR output
        cleaned_text = clean_ocr_text(normalized_text)

        extracted_pages_data.append({
            "page_num": page_num,
            "lang": lang, # Store the language used for OCR
            "text": cleaned_text
        })

    return extracted_pages_data # Return the list of dictionaries


# --- Execution ---
pdf_file_path = './data/bangla-text.pdf'
# The output file is now for structured page data, not just raw text
output_json_file = 'extracted_pages_data.json'
segmented_output_json_file = 'segmented_document_data.json'


print("Starting OCR text extraction and structuring...")
extracted_data = extract_text_with_ocr(pdf_file_path, dpi=300, lang='ben')

# Save the extracted structured page data to a JSON file
try:
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    print(f"Extracted structured page data saved to: {output_json_file}")
    print(f"Total pages extracted: {len(extracted_data)}")
except Exception as e:
    print(f"Error saving extracted_pages_data.json: {e}")

print("\n--- Starting Document Segmentation ---")
# Use the DocumentProcessor with the extracted structured data
processor = DocumentProcessor(extracted_data)
segmented_data = processor.segment_document_by_type()

# Save the segmented data to a JSON file
try:
    with open(segmented_output_json_file, 'w', encoding='utf-8') as f:
        json.dump(segmented_data, f, ensure_ascii=False, indent=2)
    print(f"Segmented document data saved to: {segmented_output_json_file}")
except Exception as e:
    print(f"Error saving segmented_document_data.json: {e}")


print("\nOCR extraction, structuring, and segmentation complete!")