import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import json 

def load_chunks_from_file(file_path="structured_chunks.json"):
    """
    Load chunks from structured_chunks.json, preserving all content and metadata.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks_with_metadata = json.load(f)
        print(f" Loaded {len(chunks_with_metadata)} chunks with metadata from '{file_path}'")
        return chunks_with_metadata
    except FileNotFoundError:
        print(f" Error: '{file_path}' not found. Please ensure the chunking process has run and the file exists.")
        return []
    except json.JSONDecodeError:
        print(f" Error: Could not decode JSON from '{file_path}'. Check file format.")
        return []

def preprocess_for_embedding(text):
    """Preprocess text for E5 embedding"""
    return f"passage: {text}"

def create_embeddings_and_index():
    """Create embeddings and FAISS index"""
    
    # Load the embedding model
    print("Loading embedding model...")
    model_name = "intfloat/multilingual-e5-base"
    model = SentenceTransformer(model_name)
    
    # Load chunks (now includes metadata)
    print("Loading chunks...")
    chunks_with_metadata = load_chunks_from_file()
    
    if not chunks_with_metadata:
        print("No chunks loaded. Exiting embedding creation.")
        return
        
    # Extract only the content for embedding, but keep original structure for saving
    chunk_contents = [chunk['content'] for chunk in chunks_with_metadata]
    
    print(f"Loaded {len(chunk_contents)} text contents for embedding.")
    
    # Create embeddings
    print("Creating embeddings...")
    preprocessed_chunks = [preprocess_for_embedding(content) for content in chunk_contents]
    embeddings = model.encode(preprocessed_chunks, normalize_embeddings=True, show_progress_bar=True)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
    index.add(embeddings.astype(np.float32))
    
    # Save index
    print("Saving FAISS index...")
    faiss.write_index(index, "faiss_index.index")
    
    # Save chunks (now saving the full chunk objects with metadata)
    print("Saving chunks with metadata...")
    with open("chunks_with_metadata.pkl", "wb") as f:
        pickle.dump(chunks_with_metadata, f)
    
    print(" Index and chunks created successfully!")
    print(f" Index contains {index.ntotal} vectors")
    print(f" Vector dimension: {dimension}")

if __name__ == "__main__":
    create_embeddings_and_index()
