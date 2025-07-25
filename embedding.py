import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks_from_file():
    """Load chunks from chunk_output.txt, extracting only the actual text content."""
    chunks = []
    with open("chunk_output.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # Split by the main chunk separator "=== CHUNK"
    chunk_sections = content.split("=== CHUNK")

    for section in chunk_sections[1:]:  # Skip the first empty section before the first "=== CHUNK"
        # Each section starts with " N ===\n[Length: ...]\n[Words: ...]\n[Lines: ...]\n------------------------------\n<ACTUAL TEXT>"
        
        # Find the start of the actual content, which is after the "------------------------------" line.
        # Split by the first occurrence of "------------------------------\n"
        parts = section.split("------------------------------\n", 1)
        
        if len(parts) > 1:
            actual_text_content = parts[1].strip()
            
            # The actual text content might also have the "================" at the end.
            # Remove the the "================" if it's there at the very end
            if actual_text_content.endswith("=================================================="):
                actual_text_content = actual_text_content[:-len("==================================================")].strip()
            
            if actual_text_content: # Ensure it's not empty after stripping
                chunks.append(actual_text_content)
    return chunks

def preprocess_for_embedding(text):
    """Preprocess text for E5 embedding"""
    return f"passage: {text}"

def create_embeddings_and_index():
    """Create embeddings and FAISS index"""
    
    # Load the embedding model
    print("Loading embedding model...")
    model_name = "intfloat/multilingual-e5-base"
    model = SentenceTransformer(model_name)
    
    # Load chunks
    print("Loading chunks...")
    chunks = load_chunks_from_file() # This will now load cleaner chunks
    print(f"Loaded {len(chunks)} chunks")
    
    # Create embeddings
    print("Creating embeddings...")
    preprocessed_chunks = [preprocess_for_embedding(chunk) for chunk in chunks]
    embeddings = model.encode(preprocessed_chunks, normalize_embeddings=True, show_progress_bar=True)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
    index.add(embeddings.astype(np.float32))
    
    # Save index
    print("Saving FAISS index...")
    faiss.write_index(index, "faiss_index.index")
    
    # Save chunks (these are the clean chunks now)
    print("Saving chunks...")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("✅ Index and chunks created successfully!")
    print(f"📊 Index contains {index.ntotal} vectors")
    print(f"📏 Vector dimension: {dimension}")

if __name__ == "__main__":
    create_embeddings_and_index()