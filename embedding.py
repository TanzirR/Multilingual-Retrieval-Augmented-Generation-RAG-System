import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the chunks from chunk_output.txt
def load_chunks_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split using the known separator pattern
    raw_chunks = content.split("==================================================")
    cleaned_chunks = [chunk.strip().split('\n', 1)[1] if '\n' in chunk else chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return cleaned_chunks

# Load chunks
chunk_file = "chunk_output.txt"
if not os.path.exists(chunk_file):
    print("❌ chunk_output.txt not found. Please run your chunking script first.")
    exit(1)

chunks = load_chunks_from_file(chunk_file)
print(f"✅ Loaded {len(chunks)} chunks.")

# Load embedding model
model_name = "intfloat/multilingual-e5-base"
print(f"🔍 Loading embedding model: {model_name}")
model = SentenceTransformer(model_name)

# Preprocess input for e5 models (if required)
def preprocess(text):
    return f"passage: {text}"

# Generate embeddings
print("⚙️ Generating embeddings...")
embeddings = model.encode([preprocess(chunk) for chunk in chunks], show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# Save embeddings and chunks with FAISS
print("📦 Saving embeddings and chunks...")

# FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for normalized vectors (cosine similarity)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.index")

# Save chunk metadata
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Embeddings and index saved successfully.")
