import re
import json
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# --- Your existing utility functions (keep them as they are) ---
def clean_chunk_lines(chunk_lines):
    cleaned = []
    for line in chunk_lines:
        original_line = line
        line = line.strip()

        if re.match(r'^[\d\s\-\,\.\?\!\u09E6-\u09EF]*$', line):
            continue
        if len(line) < 3:
            continue
        if len(re.sub(r'[^\w\u0980-\u09FF]', '', line)) < 2:
            continue
        cleaned.append(original_line)
    return cleaned

def preprocess_text(text):
    lines = text.split('\n')
    cleaned = clean_chunk_lines(lines)
    cleaned_text = '\n'.join(cleaned)
    
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' +\n', '\n', cleaned_text)
    
    return cleaned_text

def validate_chunks(chunks_content_only):
    issues = []
    
    mid_sentence_issues = detect_mid_sentence_splits(chunks_content_only)
    if mid_sentence_issues:
        issues.append(f"Found {len(mid_sentence_issues)} mid-sentence splits")
    
    too_small = [i for i, chunk in enumerate(chunks_content_only) if len(chunk) < 50]
    too_large = [i for i, chunk in enumerate(chunks_content_only) if len(chunk) > 2000]
    
    if too_small:
        issues.append(f"Found {len(too_small)} chunks that are too small (< 50 chars)")
    if too_large:
        issues.append(f"Found {len(too_large)} chunks that are too large (> 2000 chars)")
    
    empty_chunks = [i for i, chunk in enumerate(chunks_content_only) if not chunk.strip()]
    if empty_chunks:
        issues.append(f"Found {len(empty_chunks)} empty chunks")
    
    return issues

# Helper functions for `validate_chunks`
def check_chunk_continuity(chunks):
    pass

def detect_mid_sentence_splits(chunks):
    issues = []
    # Expanded fragment_starts to better detect mid-sentence breaks
    fragment_starts = (
        "বা ", "ও ", "এবং ", "তবে ", "কিন্তু ", "যে ", "এই ", "সেই ",
        "তার ", "যার ", "করে ", "হয়ে ", "থেকে ", "আর ", "যদিও ", "যখন ", "যেহেতু ",
        "কারণ ", "ফলে ", "সুতরাং ", "তারপর ", "তখন ", "তথা ", "যথা ",
        "and ", "but ", "or ", "then ", "however ", "which ", "that ", "because ", "so "
    )
    for i in range(len(chunks)-1):
        # Check if the current chunk ends without sentence punctuation and the next one starts with a fragment
        current_chunk_end = chunks[i].strip()
        if not current_chunk_end.endswith(('।', '.', '!', '?')) and len(current_chunk_end) > 0: # Ensure it doesn't end properly
            start_next = chunks[i+1].strip().split('\n')[0].lower() # Check the beginning of the next chunk
            if start_next.startswith(fragment_starts):
                issues.append((i, i+1, current_chunk_end[-50:], start_next[:50])) # Show last 50 chars of current, first 50 of next
    return issues

def merge_chunks_if_broken(chunks):
    pass

def merge_fragmented_chunks(chunks):
    pass
# --- End of existing utility functions ---

# --- NEW: Structural Segmentation and Chunking ---

def preprocess_text_for_segment_detection(text):
    """
    A lighter preprocessing specifically for segment detection.
    Removes common header/footer lines that might interfere with pattern matching,
    but keeps enough context for heuristics.
    """
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if len(stripped_line) < 2 and re.match(r'^[\d\s]*$', stripped_line):
            continue
        if "অনলাইন ব্যাচ" in stripped_line or "কলকরো ৬" in stripped_line:
            continue
        if re.match(r'^[\s\]\d\?\-–—]*$', stripped_line) and len(stripped_line) < 5:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def segment_document_by_type(pages_data):
    """
    Analyzes page data to identify logical story sections, question sections,
    and vocabulary/notes sections.
    This is a heuristic-based function and might need significant fine-tuning
    based on the exact patterns in your PDF.
    """
    segments = []
    current_segment_type = None  # 'story', 'questions', 'vocabulary'
    current_segment_text = ""
    current_segment_start_page = 0
    story_counter = 0 # To track unique story IDs
    vocabulary_counter = 0 # To track unique vocabulary sections

    for i, page_data in enumerate(pages_data):
        page_num = page_data['page_num']
        page_text = page_data['text']

        cleaned_page_text_for_detection = preprocess_text_for_segment_detection(page_text)

        is_vocabulary_page = bool(re.search(r'শব্দার্থ\s*ও\s*টীকা|মূল\s*শব্দ\s*শব্দের\s*অর্থ\s*ও\s*ব্যাখ্যা', cleaned_page_text_for_detection, re.MULTILINE))

        pattern_mcq_like = r'(?m)^\s*(\d+[\u09E6-\u09EF]*|[কখগঘ]|[a-zA-Z])\s*[।\.]?\s*.*?\n\s*(\([কখগঘa-zA-Z]\)|\([a-zA-Z]\))'
        pattern_long_q_like = r'(?m)^\s*(পাঠ্যপুস্তকের\s*প্রশ্ন|বহুনির্বাচনী|উদ্দীপক|প্রশ্নের\s*উত্তর|দীর্ঘ\s*প্রশ্নাবলি)' # Added common question headings
        is_question_page = bool(re.search(pattern_mcq_like, cleaned_page_text_for_detection)) or \
                           bool(re.search(pattern_long_q_like, cleaned_page_text_for_detection))

        is_story_page = bool(re.search(r'মূল\s*গল্প|অনুপমের\s*বযস\s*সাতাশ', cleaned_page_text_for_detection))
        # Heuristic for first page if it doesn't match other types
        if i == 0 and not (is_question_page or is_vocabulary_page):
            is_story_page = True

        new_segment_type = None
        # Prioritize question and vocabulary pages as they have distinct formats
        if is_question_page:
            new_segment_type = 'questions'
        elif is_vocabulary_page:
            new_segment_type = 'vocabulary'
        elif is_story_page:
            new_segment_type = 'story'
        else:
            # If no specific type detected, assume it continues the previous type or is unclassified
            if current_segment_type:
                new_segment_type = current_segment_type
            else:
                new_segment_type = 'unclassified' # Should ideally not happen often

        if new_segment_type != current_segment_type and current_segment_type is not None:
            segment_id = ""
            if current_segment_type == 'story':
                story_counter += 1
                segment_id = f"story_{story_counter}"
            elif current_segment_type == 'questions':
                # Link questions to the *previous* story segment
                segment_id = f"questions_for_story_{story_counter}"
            elif current_segment_type == 'vocabulary':
                vocabulary_counter += 1
                segment_id = f"vocabulary_{vocabulary_counter}"
            elif current_segment_type == 'unclassified':
                segment_id = f"unclassified_segment_{len(segments) + 1}"

            segments.append({
                "type": current_segment_type,
                "id": segment_id,
                "start_page": current_segment_start_page,
                "end_page": page_num - 1,
                "text": current_segment_text.strip()
            })
            
            current_segment_text = page_text
            current_segment_start_page = page_num
            current_segment_type = new_segment_type
        else:
            if current_segment_text:
                current_segment_text += "\n" + page_text
            else:
                current_segment_text = page_text
                current_segment_start_page = page_num
                current_segment_type = new_segment_type
    
    # Finalize the last segment
    if current_segment_text:
        segment_id = ""
        if current_segment_type == 'story':
            story_counter += 1
            segment_id = f"story_{story_counter}"
        elif current_segment_type == 'questions':
            # Link questions to the *last* story segment encountered
            segment_id = f"questions_for_story_{story_counter}"
        elif current_segment_type == 'vocabulary':
            vocabulary_counter += 1
            segment_id = f"vocabulary_{vocabulary_counter}"
        elif current_segment_type == 'unclassified':
            segment_id = f"unclassified_segment_{len(segments) + 1}"

        segments.append({
            "type": current_segment_type,
            "id": segment_id,
            "start_page": current_segment_start_page,
            "end_page": pages_data[-1]['page_num'],
            "text": current_segment_text.strip()
        })
    
    segments = [s for s in segments if s['text'].strip()]
    
    return segments

def chunk_segment(segment_type, segment_text, segment_id, page_range):
    chunks_with_metadata = []

    segment_text = preprocess_text(segment_text)

    # --- Fine-tuning Suggestion 1: Enhance common_separators ---
    # Add an empty string "" as the last resort separator. This ensures that
    # if no other separators are found, the text will be split character by character,
    # preventing excessively large chunks that cannot be broken otherwise.
    common_separators = ["।", ".", "!", "?", "\n\n", "\n", "—", ",", " ", ""]

    if segment_type == 'story':
        story_splitter = RecursiveCharacterTextSplitter(
            separators=common_separators,
            chunk_size=700,
            chunk_overlap=200,
            length_function=len
        )
        sub_chunks = story_splitter.split_text(segment_text)
        for i, sub_chunk in enumerate(sub_chunks):
            if sub_chunk.strip():
                chunks_with_metadata.append({
                    "content": sub_chunk.strip(),
                    "metadata": {
                        "type": "story_segment",
                        "segment_id": segment_id,
                        "chunk_idx": i,
                        "page_range": f"{page_range[0]}-{page_range[1]}",
                        "source": "textbook_story"
                    }
                })

    elif segment_type == 'questions':
        question_pattern = re.compile(
            r'(?m)^(\s*(\d+[\u09E6-\u09EF]*|[কখগঘ]|[a-zA-Z])\s*[।\.]?\s*.*?)'
            r'(\n\s*(\([কখগঘa-zA-Z]\)\s*.*?)+)?'
            r'(?=\n\s*(\d+[\u09E6-\u09EF]*|[কখগঘ]|[a-zA-Z])\s*[।\.]?|\Z)',
            re.DOTALL
        )

        # Handle introduction text before questions
        intro_text_match = re.match(r'^(.*?)(?=\n*\s*(?:\d+[\u09E6-\u09EF]*|[কখগঘ]|[a-zA-Z])\s*[।\.]?)', segment_text, re.DOTALL)
        if intro_text_match and intro_text_match.group(1).strip():
            intro_text = intro_text_match.group(1).strip()
            if len(intro_text) > 100: # Increase threshold for intro text chunking
                chunks_with_metadata.append({
                    "content": intro_text,
                    "metadata": {
                        "type": "question_introduction",
                        "segment_id": segment_id,
                        "chunk_idx": 0,
                        "page_range": f"{page_range[0]}-{page_range[1]}",
                        "source": "textbook_question_intro"
                    }
                })
                segment_text = segment_text[len(intro_text_match.group(0)):].strip()

        # Handle passages related to questions
        passage_for_q_match = re.search(r'^(.*?)(?=\n*\s*(\d+[\u09E6-\u09EF]*|[কখগঘ]|[a-zA-Z])\s*[।\.]?|\Z)', segment_text, re.DOTALL)
        if passage_for_q_match and passage_for_q_match.group(1).strip() and not re.match(r'^\s*(\d+[\u09E6-\u09EF]*|[কখগঘ]|[a-zA-Z])', passage_for_q_match.group(1).strip()):
            passage_text = passage_for_q_match.group(1).strip()
            if len(passage_text) > 100: # Increase threshold for passage text chunking
                chunks_with_metadata.append({
                    "content": passage_text,
                    "metadata": {
                        "type": "question_passage",
                        "segment_id": segment_id,
                        "chunk_idx": len(chunks_with_metadata),
                        "page_range": f"{page_range[0]}-{page_range[1]}",
                        "source": "textbook_question_passage"
                    }
                })
                segment_text = segment_text[len(passage_for_q_match.group(0)):].strip()

        question_idx_counter = 0
        for match in question_pattern.finditer(segment_text):
            full_question_text = match.group(0).strip()
            if not full_question_text:
                continue

            question_num_match = re.match(r'^\s*(\d+[\u09E6-\u09EF]*)', full_question_text)
            question_num = question_num_match.group(1).strip() if question_num_match else "N/A"

            chunks_with_metadata.append({
                "content": full_question_text,
                "metadata": {
                    "type": "question",
                    "segment_id": segment_id,
                    "question_number": question_num,
                    "chunk_idx": question_idx_counter,
                    "page_range": f"{page_range[0]}-{page_range[1]}",
                    "source": "textbook_question"
                }
            })
            question_idx_counter += 1

        # Fallback for any remaining unparsed text in question section
        if not chunks_with_metadata and segment_text.strip():
            general_splitter = RecursiveCharacterTextSplitter(
                separators=common_separators,
                chunk_size=700,
                chunk_overlap=200
            )
            general_chunks = general_splitter.split_text(segment_text)
            for i, gc in enumerate(general_chunks):
                if gc.strip():
                    chunks_with_metadata.append({
                        "content": gc.strip(),
                        "metadata": {
                            "type": "unclassified_question_text",
                            "segment_id": segment_id,
                            "chunk_idx": i,
                            "page_range": f"{page_range[0]}-{page_range[1]}",
                            "source": "textbook_question_general"
                        }
                    })

    elif segment_type == 'vocabulary':
        vocabulary_entry_pattern = re.compile(
            r'^\s*([^\n:]+?)\s*[:—]\s*(.*?)(?=\n\s*[^\n:]+?\s*[:—]|\Z)',
            re.MULTILINE | re.DOTALL
        )
        line_by_line_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ""], # Added "" here too for safety
            chunk_size=150,
            chunk_overlap=0
        )

        vocab_idx_counter = 0
        found_specific_entries = False

        for match in vocabulary_entry_pattern.finditer(segment_text):
            key_term = match.group(1).strip()
            definition = match.group(2).strip()

            if key_term and definition:
                full_entry = f"{key_term}: {definition}"
                chunks_with_metadata.append({
                    "content": full_entry,
                    "metadata": {
                        "type": "vocabulary_entry",
                        "segment_id": segment_id,
                        "key_term": key_term,
                        "chunk_idx": vocab_idx_counter,
                        "page_range": f"{page_range[0]}-{page_range[1]}",
                        "source": "textbook_vocabulary_structured"
                    }
                })
                vocab_idx_counter += 1
                found_specific_entries = True

        if not found_specific_entries or vocab_idx_counter == 0:
            print(f"DEBUG: Falling back to line-by-line chunking for vocabulary segment '{segment_id}'.")
            general_vocab_chunks = line_by_line_splitter.split_text(segment_text)
            for i, gc in enumerate(general_vocab_chunks):
                if gc.strip():
                    chunks_with_metadata.append({
                        "content": gc.strip(),
                        "metadata": {
                            "type": "vocabulary_entry_general",
                            "segment_id": segment_id,
                            "key_term": "N/A",
                            "chunk_idx": i,
                            "page_range": f"{page_range[0]}-{page_range[1]}",
                            "source": "textbook_vocabulary_general"
                        }
                    })
    
    # --- Fine-tuning Suggestion 2: Hard Re-chunking for excessively large chunks ---
    # This ensures that no chunk, even after initial splitting, exceeds a strict maximum size.
    final_post_processed_chunks = []
    MAX_HARD_CHUNK_SIZE = 1800 # Adjusted maximum chunk size to be below 2000

    for chunk_item in chunks_with_metadata:
        if len(chunk_item['content']) > MAX_HARD_CHUNK_SIZE:
            print(f"DEBUG: Re-chunking excessively large chunk (size: {len(chunk_item['content'])} chars) from segment '{segment_id}' type '{chunk_item['metadata']['type']}'")
            # Use CharacterTextSplitter for a brute-force split to respect max size
            hard_splitter = CharacterTextSplitter(
                chunk_size=MAX_HARD_CHUNK_SIZE - 100, # Chunk into pieces slightly smaller than MAX_HARD_CHUNK_SIZE
                chunk_overlap=50, # Maintain some overlap for re-chunked parts
                separator="" # Force character-level split if needed
            )
            sub_chunks = hard_splitter.split_text(chunk_item['content'])
            
            for j, sub_chunk in enumerate(sub_chunks):
                if sub_chunk.strip():
                    # Create new metadata for re-chunked parts
                    new_metadata = {**chunk_item['metadata']}
                    new_metadata['type'] = new_metadata['type'] + "_rechunked" # Indicate it was re-chunked
                    new_metadata['chunk_idx'] = f"{chunk_item['metadata']['chunk_idx']}_{j}" # New index for sub-chunk
                    final_post_processed_chunks.append({
                        "content": sub_chunk.strip(),
                        "metadata": new_metadata
                    })
        else:
            final_post_processed_chunks.append(chunk_item)
    
    chunks_with_metadata = final_post_processed_chunks # Update the list for subsequent filtering
    
    # Post-processing: Filter out very small chunks or merge them
    final_chunks = []
    i = 0
    while i < len(chunks_with_metadata):
        chunk = chunks_with_metadata[i]
        if len(chunk['content']) < 50 and i < len(chunks_with_metadata) - 1:
            next_chunk = chunks_with_metadata[i+1]
            # Ensure they are from the same segment and type to merge logically
            if chunk['metadata']['segment_id'] == next_chunk['metadata']['segment_id'] and \
               chunk['metadata']['type'].replace("_rechunked", "") == next_chunk['metadata']['type'].replace("_rechunked", ""): # Compare original type
                merged_content = chunk['content'] + "\n" + next_chunk['content']
                merged_chunk = {
                    "content": merged_content,
                    "metadata": {
                        **chunk['metadata'],
                        "chunk_idx": chunk['metadata']['chunk_idx'],
                    }
                }
                final_chunks.append(merged_chunk)
                i += 2
            else:
                if len(chunk['content']) >= 20:
                    final_chunks.append(chunk)
                i += 1
        else:
            final_chunks.append(chunk)
            i += 1

    return final_chunks


def save_chunks_with_metadata_to_json(chunks_data, output_file='structured_chunks.json'):
    """
    Saves chunks with comprehensive metadata to a JSON file.
    Each item in chunks_data is expected to be a dictionary with 'content' and 'metadata'.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"Structured chunks saved to: '{output_file}'")
        
        total_chunks = len(chunks_data)
        if total_chunks > 0:
            total_chars = sum(len(chunk['content']) for chunk in chunks_data)
            avg_length = total_chars / total_chunks
            min_length = min(len(chunk['content']) for chunk in chunks_data)
            max_length = max(len(chunk['content']) for chunk in chunks_data)
            print(f"   Final Statistics:")
            print(f"   Total chunks: {total_chunks}")
            print(f"   Average length: {avg_length:.0f} characters")
            print(f"   Size range: {min_length} - {max_length} characters")
            print(f"   Total content: {total_chars} characters")
        else:
            print(" No chunks generated.")

        return True
            
    except Exception as e:
        print(f" Error saving structured chunks: {e}")
        return False

# --- Main pipeline function ---
def main():
    """Main chunking pipeline with structural segmentation and metadata."""
    
    try:
        # Load the structured extracted text from ocr_pdf_dynamic.py output
        with open('extracted_pages_data.json', 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
        print(" Structured page data loaded from 'extracted_pages_data.json'")
    except FileNotFoundError:
        print(" 'extracted_pages_data.json' not found. Please run ocr_pdf_dynamic.py first.")
        return []
    except json.JSONDecodeError:
        print(" Error decoding 'extracted_pages_data.json'. Ensure it's valid JSON.")
        return []
    
    print(f" Processing {len(pages_data)} pages.")

    # Step 1: Segment the document into logical story and question blocks
    print("\n Segmenting document into story and question blocks...")
    segments = segment_document_by_type(pages_data)
    print(f" Document segmented into {len(segments)} logical blocks.")
    for segment in segments:
        text_preview = segment['text'][:100].replace('\n', ' ') + '...' if len(segment['text']) > 100 else segment['text'].replace('\n', ' ')
        print(f"   - Type: {segment['type']}, ID: {segment['id']}, Pages: {segment['start_page']}-{segment['end_page']} | Content Preview: '{text_preview}'")

    all_final_chunks_with_metadata = []

    # Step 2: Chunk each segment based on its type and add metadata
    print("\n Chunking each segment with specific rules and metadata...")
    for segment in segments:
        cleaned_segment_text = preprocess_text(segment['text'])
        
        chunked_results = chunk_segment(
            segment_type=segment['type'],
            segment_text=cleaned_segment_text,
            segment_id=segment['id'],
            page_range=(segment['start_page'], segment['end_page'])
        )
        all_final_chunks_with_metadata.extend(chunked_results)
    
    print(f"Total chunks created: {len(all_final_chunks_with_metadata)}")

    # Step 3: Validate final chunks (extract content strings for validation)
    raw_contents_for_validation = [c['content'] for c in all_final_chunks_with_metadata]
    print("\n Validating final chunks...")
    validation_issues = validate_chunks(raw_contents_for_validation)
    if validation_issues:
        print(" Validation issues found:")
        for issue in validation_issues:
            print(f"   - {issue}")
    else:
        print(" All validation tests passed!")

    # Step 4: Save structured chunks to a JSON file
    print("\n Saving chunks with metadata...")
    save_chunks_with_metadata_to_json(all_final_chunks_with_metadata)
    
    # Step 5: Show sample chunk
    if all_final_chunks_with_metadata:
        print(f"\n Sample chunk (first 200 chars):")
        sample_chunk = all_final_chunks_with_metadata[0]
        print(f"--- CHUNK 1 ---")
        print(f"Type: {sample_chunk['metadata']['type']}")
        print(f"Segment ID: {sample_chunk['metadata']['segment_id']}")
        print(f"Page Range: {sample_chunk['metadata']['page_range']}")
        if sample_chunk['metadata']['type'] == 'question':
            print(f"Question No: {sample_chunk['metadata'].get('question_number', 'N/A')}")
        print(f"Content Preview: {sample_chunk['content'][:200]}...")
        print("-" * 30)
    
    return all_final_chunks_with_metadata

if __name__ == "__main__":
    final_chunks = main()