import re

def preprocess_text(text):
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_valid_bangla(text, min_words=5):
    if len(text.split()) < min_words:
        return False
    if not re.search(r'[\u0980-\u09FF]', text):  # No Bangla Unicode
        return False
    if re.fullmatch(r'[\W\d\s]+', text):  # Only symbols, digits, or empty
        return False
    return True

def split_sentences(text):
    # Split Bangla text by danda and other sentence endings, keep punctuation
    sentences = re.split(r'(?<=[।!?])\s*', text)
    return [s.strip() for s in sentences if s.strip()]

def split_and_clean_chunks(text, max_words=100, merge_min_words=10):
    text = preprocess_text(text)
    paragraphs = text.split("==================================================")
    cleaned_sentences = []

    # First: clean and split paragraphs into sentences
    for para in paragraphs:
        para = para.strip()
        if is_valid_bangla(para):
            sentences = split_sentences(para)
            for s in sentences:
                if is_valid_bangla(s, min_words=3):  # smaller min to keep shorter sentences
                    cleaned_sentences.append(s)

    final_chunks = []
    buffer = []

    def buffer_word_count(buf):
        return sum(len(s.split()) for s in buf)

    for sentence in cleaned_sentences:
        # If adding this sentence keeps buffer below max_words, add it
        if buffer_word_count(buffer) + len(sentence.split()) <= max_words:
            buffer.append(sentence)
        else:
            # Flush buffer as a chunk
            if buffer_word_count(buffer) >= merge_min_words:
                final_chunks.append(" ".join(buffer))
                buffer = [sentence]
            else:
                # Buffer too small, merge with last chunk if exists
                if final_chunks:
                    final_chunks[-1] += " " + " ".join(buffer) + " " + sentence
                    buffer = []
                else:
                    # No previous chunk, just flush buffer and start new
                    final_chunks.append(" ".join(buffer))
                    buffer = [sentence]

    # Flush remaining buffer
    if buffer:
        final_chunks.append(" ".join(buffer))

    return final_chunks

# Usage example (unchanged)
with open("bangla_ocr_output.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

chunks = split_and_clean_chunks(raw_text)

with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"Chunk {i+1}:\n{chunk}\n\n" + "="*50 + "\n\n")

print(f"Cleaned and created {len(chunks)} chunks")
