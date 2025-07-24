import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks_from_file():
    """Load chunks from chunk_output.txt"""
    chunks = []
    
    with open("chunk_output.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by chunk separators
    chunk_sections = content.split("=== CHUNK")
    
    for section in chunk_sections[1:]:  # Skip the first empty section
        # Extract chunk content (everything after the chunk number line)
        lines = section.strip().split('\n')
        if len(lines) > 1:
            # Skip the chunk number line and separator lines
            chunk_content = []
            for line in lines[1:]:
                if line.strip() and not line.startswith("===="):
                    chunk_content.append(line.strip())
            
            if chunk_content:
                chunks.append('\n'.join(chunk_content))
    
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
    chunks = load_chunks_from_file()
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
    
    # Save chunks
    print("Saving chunks...")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("✅ Index and chunks created successfully!")
    print(f"📊 Index contains {index.ntotal} vectors")
    print(f"📏 Vector dimension: {dimension}")

if __name__ == "__main__":
    create_embeddings_and_index()
