from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import json
from datetime import datetime

def clean_chunk_lines(chunk_lines):
    """Remove lines that look like standalone numbers or gibberish page markers."""
    cleaned = []
    for line in chunk_lines:
        original_line = line # Keep original for printing
        line = line.strip() # Work with stripped line for checks

        # Check 1: Consists mostly of digits, punctuation, or is empty
        if re.match(r'^[\d\s\-\,\.\?\!\u09E6-\u09EF]*$', line):
            print(f"DEBUG: Removing (digits/punc/empty): '{original_line}'")
            continue
        # Check 2: Very short lines
        if len(line) < 3:
            print(f"DEBUG: Removing (too short): '{original_line}'")
            continue
        # Check 3: Mostly punctuation
        if len(re.sub(r'[^\w\u0980-\u09FF]', '', line)) < 2:
            print(f"DEBUG: Removing (mostly punc): '{original_line}'")
            continue
        cleaned.append(original_line) # Append original line if it passes all checks
    return cleaned

def preprocess_text(text):
    """Clean the text before chunking by removing noise lines"""
    lines = text.split('\n')
    cleaned = clean_chunk_lines(lines)
    cleaned_text = '\n'.join(cleaned)
    
    # Additional cleaning
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    # Remove trailing spaces
    cleaned_text = re.sub(r' +\n', '\n', cleaned_text)
    
    return cleaned_text

def check_chunk_continuity(chunks):
    """Check if chunks maintain proper context continuity"""
    issues = []
    for i in range(len(chunks)-1):
        tail = chunks[i].strip().split('\n')[-1]
        head = chunks[i+1].strip().split('\n')[0]
        
        print(f"CHUNK {i+1} ➡ CHUNK {i+2}")
        print(f"END: {tail}")
        print(f"START: {head}")
        
        # Enhanced heuristic: If head starts with Bengali alphabet, uppercase, or dialogue marker
        if re.match(r'^[অ-হA-Z"\'—]', head):
            print("✅ Context likely maintained.\n")
        else:
            print("⚠️ Possible context cut.\n")
            issues.append((i, i+1, tail, head))
    
    return issues

def detect_mid_sentence_splits(chunks):
    """Detect chunks that start with sentence fragments"""
    issues = []
    fragment_starts = (
        "বা ", "ও ", "এবং ", "তবে ", "কিন্তু ", "যে ", "এই ", "সেই ", 
        "তার ", "যার ", "করে ", "হয়ে ", "থেকে ", "আর ", "যদিও ",
        "and ", "but ", "or ", "then ", "however ", "which ", "that "
    )
    
    for i in range(len(chunks)-1):
        end_chunk = chunks[i].strip().split('\n')[-1]
        start_next = chunks[i+1].strip().split('\n')[0]
        
        if start_next.startswith(fragment_starts):
            issues.append((i, i+1, end_chunk, start_next))
    
    return issues

def merge_chunks_if_broken(chunks):
    """Merge chunk i+1 into chunk i if continuity check fails"""
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        chunk = chunks[i].strip()
        
        if i < len(chunks) - 1:
            last_line = chunk.split('\n')[-1].strip()
            next_first_line = chunks[i+1].strip().split('\n')[0].strip()
            
            # Clean potential noise lines first
            last_line_clean = re.sub(r'[\d\s\-\,\.\?\!]*', '', last_line)
            next_first_line_clean = re.sub(r'[\d\s\-\,\.\?\!]*', '', next_first_line)
            
            # Enhanced merging conditions
            should_merge = (
                # Next chunk starts with fragment
                not re.match(r'^[অ-হA-Z"\'—]', next_first_line_clean) or
                # Current chunk ends abruptly (no sentence terminator)
                not re.search(r'[।\.!\?]$', last_line.strip()) or
                # Next chunk is very short (likely incomplete)
                len(chunks[i+1].strip()) < 100
            )
            
            if should_merge:
                merged_chunk = chunk + "\n" + chunks[i+1].strip()
                merged_chunks.append(merged_chunk)
                i += 2  # Skip next chunk since merged
                continue
        
        merged_chunks.append(chunk)
        i += 1
    
    return merged_chunks

def merge_fragmented_chunks(chunks):
    """Merge chunks that start with connecting words or fragments"""
    merged_chunks = []
    buffer = chunks[0]

    fragment_starts = (
        "বা ", "ও ", "এবং ", "তবে ", "কিন্তু ", "আর ", "যদিও ", 
        "যে ", "এই ", "সেই ", "তার ", "যার ", "করে ", "হয়ে ", "থেকে ",
        "and ", "but ", "or ", "then ", "however ", "which ", "that "
    )

    for i in range(1, len(chunks)):
        start_line = chunks[i].strip().split('\n')[0]
        
        if start_line.startswith(fragment_starts):
            buffer += " " + chunks[i]
        else:
            merged_chunks.append(buffer)
            buffer = chunks[i]

    merged_chunks.append(buffer)
    return merged_chunks

def chunk_text_enhanced(text, chunk_size=300, chunk_overlap=150, preserve_dialogue=True):
    """Enhanced chunking with better sentence awareness"""
    
    # Preserve dialogue and poetry formatting
    if preserve_dialogue:
        separators = ["।", ".", "!", "?", "\n\n"]  # Avoid breaking on single newlines
    else:
        separators = ["\n\n", "\n", "।", ".", "!", "?"]
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def validate_chunks(chunks):
    """Unit testing for chunk quality"""
    issues = []
    
    # Test 1: No mid-sentence splits
    mid_sentence_issues = detect_mid_sentence_splits(chunks)
    if mid_sentence_issues:
        issues.append(f"Found {len(mid_sentence_issues)} mid-sentence splits")
    
    # Test 2: Reasonable chunk sizes
    too_small = [i for i, chunk in enumerate(chunks) if len(chunk) < 50]
    too_large = [i for i, chunk in enumerate(chunks) if len(chunk) > 2000]
    
    if too_small:
        issues.append(f"Found {len(too_small)} chunks that are too small (< 50 chars)")
    if too_large:
        issues.append(f"Found {len(too_large)} chunks that are too large (> 2000 chars)")
    
    # Test 3: No empty chunks
    empty_chunks = [i for i, chunk in enumerate(chunks) if not chunk.strip()]
    if empty_chunks:
        issues.append(f"Found {len(empty_chunks)} empty chunks")
    
    return issues

def save_chunks_with_metadata(chunks, output_file='chunk_output.txt', metadata_file='chunk_metadata.json'):
    """Save chunks with comprehensive metadata"""
    
    # Calculate statistics
    chunk_stats = {
        'total_chunks': len(chunks),
        'avg_length': sum(len(chunk) for chunk in chunks) / len(chunks),
        'min_length': min(len(chunk) for chunk in chunks),
        'max_length': max(len(chunk) for chunk in chunks),
        'total_characters': sum(len(chunk) for chunk in chunks),
        'processing_date': datetime.now().isoformat()
    }
    
    # Save chunks with metadata
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i+1} ===\n")
                f.write(f"[Length: {len(chunk)} chars]\n")
                f.write(f"[Words: ~{len(chunk.split())} words]\n")
                f.write(f"[Lines: {len(chunk.split(chr(10)))} lines]\n")
                f.write("-" * 30 + "\n")
                f.write(chunk)
                f.write(f"\n\n{'='*50}\n\n")
        
        print(f"✅ Chunks saved to: {output_file}")
        
        # Save metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Metadata saved to: {metadata_file}")
        return chunk_stats
        
    except Exception as e:
        print(f"❌ Error saving chunks: {e}")
        return None

def main():
    """Main chunking pipeline with all enhancements"""
    
    # Load the extracted text
    try:
        with open('extracted_output.txt', 'r', encoding='utf-8') as f:
            knowledge_base_text = f.read()
        print("✅ Text loaded from extracted_output.txt")
    except FileNotFoundError:
        print("❌ extracted_output.txt not found. Please run extract.py first.")
        return
    
    print(f"📄 Original text length: {len(knowledge_base_text)} characters")
    
    # Step 1: Preprocess text (NEW)
    print("\n🔧 Preprocessing text...")
    cleaned_text = preprocess_text(knowledge_base_text)
    print(f"✅ Text cleaned: {len(knowledge_base_text)} → {len(cleaned_text)} characters")
    
    # Step 2: Enhanced chunking
    print("\n📝 Creating initial chunks...")
    text_chunks = chunk_text_enhanced(cleaned_text, preserve_dialogue=True)
    print(f"✅ Text divided into {len(text_chunks)} chunks.")
    
    # Step 3: Check continuity
    print("\n📋 Checking chunk continuity...")
    continuity_issues = check_chunk_continuity(text_chunks)
    
    # Step 4: Detect mid-sentence splits
    print("\n🔍 Detecting mid-sentence splits...")
    split_issues = detect_mid_sentence_splits(text_chunks)
    if split_issues:
        print(f"Found {len(split_issues)} potential mid-sentence splits:")
        for i1, i2, end_c, start_c in split_issues[:3]:  # Show first 3
            print(f"  CHUNK {i1+1} → CHUNK {i2+1}")
            print(f"  End: {end_c[:50]}...")
            print(f"  Start: {start_c[:50]}...")
    else:
        print("✅ No mid-sentence splits detected!")
    
    # Step 5: Merge broken chunks
    print("\n🔧 Merging broken chunks...")
    merged_chunks = merge_chunks_if_broken(text_chunks)
    print(f"✅ Merged broken chunks: {len(text_chunks)} → {len(merged_chunks)}")
    
    # Step 6: Merge fragmented chunks
    print("\n🔧 Merging fragmented chunks...")
    final_chunks = merge_fragmented_chunks(merged_chunks)
    print(f"✅ Final merge: {len(merged_chunks)} → {len(final_chunks)}")
    
    # Step 7: Validate final chunks (NEW)
    print("\n✅ Validating final chunks...")
    validation_issues = validate_chunks(final_chunks)
    if validation_issues:
        print("⚠️ Validation issues found:")
        for issue in validation_issues:
            print(f"  - {issue}")
    else:
        print("✅ All validation tests passed!")
    
    # Step 8: Save with metadata (ENHANCED)
    print("\n💾 Saving chunks with metadata...")
    stats = save_chunks_with_metadata(final_chunks)
    
    if stats:
        print(f"\n📊 Final Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Average length: {stats['avg_length']:.0f} characters")
        print(f"  Size range: {stats['min_length']} - {stats['max_length']} characters")
        print(f"  Total content: {stats['total_characters']} characters")
    
    # Step 9: Show sample chunk
    if final_chunks:
        print(f"\n📄 Sample chunk (first 200 chars):")
        print(f"'{final_chunks[0][:200]}...'")
    
    return final_chunks

if __name__ == "__main__":
    final_chunks = main()