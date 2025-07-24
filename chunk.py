from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def check_chunk_continuity(chunks):
    for i in range(len(chunks)-1):
        tail = chunks[i].strip().split('\n')[-1]
        head = chunks[i+1].strip().split('\n')[0]
        print(f"CHUNK {i+1} ➡ CHUNK {i+2}")
        print(f"END: {tail}")
        print(f"START: {head}")
        # Simple heuristic: If head starts with Bengali alphabet or uppercase, OK; else warn
        if re.match(r'^[অ-হA-Z]', head):
            print("✅ Context likely maintained.\n")
        else:
            print("⚠️ Possible context cut.\n")

def detect_mid_sentence_splits(chunks):
    issues = []
    for i in range(len(chunks)-1):
        end_chunk = chunks[i].strip().split('\n')[-1]
        start_next = chunks[i+1].strip().split('\n')[0]
        # If start of next chunk looks like sentence fragment (e.g. starts with conjunction or phrase)
        if start_next.startswith(("বা ", "ও ", "এবং ", "তবে ", "কিন্তু ", "যে ", "এই ", "সেই ", "তার ", "যার ", "করে ", "হয়ে ", "থেকে ")):
            issues.append((i, i+1, end_chunk, start_next))
    return issues

def clean_chunk_lines(chunk_lines):
    """Remove lines that look like standalone numbers or gibberish page markers."""
    cleaned = []
    for line in chunk_lines:
        # Remove line if it consists mostly of digits, punctuation, or is empty
        if re.match(r'^[\d\s\-\,\.\?\!\u09E6-\u09EF]*$', line.strip()):
            continue
        cleaned.append(line)
    return cleaned

def merge_chunks_if_broken(chunks):
    """Merge chunk i+1 into chunk i if continuity check fails."""
    merged_chunks = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i].strip()
        if i < len(chunks) - 1:
            # Extract last line of current chunk, first line of next chunk
            last_line = chunk.split('\n')[-1].strip()
            next_first_line = chunks[i+1].strip().split('\n')[0].strip()
            
            # Clean potential noise lines first
            last_line_clean = re.sub(r'[\d\s\-\,\.\?\!]*', '', last_line)
            next_first_line_clean = re.sub(r'[\d\s\-\,\.\?\!]*', '', next_first_line)
            
            # Heuristic: if next chunk first line doesn't start with Bengali letter or uppercase letter, merge
            if not re.match(r'^[অ-হA-Z]', next_first_line_clean):
                # Merge chunks[i] and chunks[i+1]
                merged_chunk = chunk + "\n" + chunks[i+1].strip()
                merged_chunks.append(merged_chunk)
                i += 2  # skip next chunk since merged
                continue
        
        merged_chunks.append(chunk)
        i += 1
    return merged_chunks

def merge_fragmented_chunks(chunks):
    merged_chunks = []
    buffer = chunks[0]

    # Words or phrases that often indicate a mid-sentence fragment start in Bengali (add more if needed)
    fragment_starts = ("বা ", "ও ", "এবং ", "তবে ", "কিন্তু ", "আর ", "যদিও ", "যে ", "এই ", "সেই ", "তার ", "যার ", "করে ", "হয়ে ", "থেকে ")

    for i in range(1, len(chunks)):
        start_line = chunks[i].strip().split('\n')[0]
        # Check if next chunk starts with a fragment phrase
        if start_line.startswith(fragment_starts):
            # Merge with current buffer chunk (add a space/newline if needed)
            buffer += " " + chunks[i]
        else:
            # Current chunk finished, add buffer and reset
            merged_chunks.append(buffer)
            buffer = chunks[i]

    # Add the last buffer
    merged_chunks.append(buffer)
    return merged_chunks

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Chunks text using sentence-aware splitting with Bengali/English support."""
    
    # Updated separators to preserve natural sentence boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "।", ".", "!", "?"],  # Bengali + English sentence markers
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

# --- Execution ---
# Load the extracted text from file
try:
    with open('extracted_output.txt', 'r', encoding='utf-8') as f:
        knowledge_base_text = f.read()
    print("✅ Text loaded from extracted_output.txt")
except FileNotFoundError:
    print("❌ extracted_output.txt not found. Please run extract.py first.")
    exit(1)

text_chunks = chunk_text(knowledge_base_text)

print(f"✅ Text divided into {len(text_chunks)} chunks.")

# Check chunk continuity
print("\n📋 Checking chunk continuity...")
check_chunk_continuity(text_chunks)

# Detect mid-sentence splits
print("\n🔍 Detecting mid-sentence splits...")
issues = detect_mid_sentence_splits(text_chunks)
if issues:
    print(f"Found {len(issues)} potential mid-sentence splits:")
    for i1, i2, end_c, start_c in issues:
        print(f"Potential mid-sentence split between CHUNK {i1+1} and CHUNK {i2+1}:")
        print(f"End of CHUNK {i1+1}: {end_c}")
        print(f"Start of CHUNK {i2+1}: {start_c}")
        print()
else:
    print("✅ No mid-sentence splits detected!")

# Apply enhanced post-processing to merge broken chunks
print("\n🔧 Applying enhanced post-processing to merge broken chunks...")
merged_chunks = merge_chunks_if_broken(text_chunks)
print(f"✅ Merged broken chunks: {len(text_chunks)} → {len(merged_chunks)}")

# Apply fragmented chunk merging
print("\n🔧 Applying fragmented chunk merging...")
final_merged_chunks = merge_fragmented_chunks(merged_chunks)
print(f"✅ Final merge: {len(merged_chunks)} → {len(final_merged_chunks)}")

# Use final merged chunks for output
final_chunks = final_merged_chunks

# Save chunks to file
output_file = 'chunk_output.txt'
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(final_chunks):
            f.write(f"=== CHUNK {i+1} ===\n")
            f.write(chunk)
            f.write(f"\n\n{'='*50}\n\n")
    
    print(f"✅ Chunks saved to: {output_file}")
    print(f"📊 Total chunks: {len(final_chunks)}")
    print(f"📄 Average chunk size: {sum(len(chunk) for chunk in final_chunks) / len(final_chunks):.0f} characters")
except Exception as e:
    print(f"❌ Error saving chunks: {e}")
