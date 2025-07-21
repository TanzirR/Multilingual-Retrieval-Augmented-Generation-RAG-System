#Step 2: Split into manageable chunks, preserving context: Paragraph-Based (with max word limit)
def split_into_chunks(text, max_words=100):
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para.strip())
        else:
            # Break long paragraph into smaller chunks
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i+max_words])
                chunks.append(chunk.strip())
    return chunks


# Read text from the saved OCR output file
with open("bangla_ocr_output.txt", "r", encoding="utf-8") as f:
    extracted_text_from_file = f.read()

paragraph_chunks = split_into_chunks(extracted_text_from_file)

with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(paragraph_chunks):
        f.write(f"Chunk {i+1}:\n{chunk}\n\n" + "="*50 + "\n\n")

print("✅ Step2: Splitting complete.")
print(f"📄 Created {len(paragraph_chunks)} chunks")