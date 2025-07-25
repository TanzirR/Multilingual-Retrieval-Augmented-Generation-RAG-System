from sentence_transformers import SentenceTransformer, util
import pickle

model = SentenceTransformer("intfloat/multilingual-e5-base")

query = "query: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?" # Query prefix for E5

# Load the chunks from the .pkl file
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Get Chunk 21 (index 20)
if len(chunks) > 20:
    target_chunk = f"passage: {chunks[20]}" # Passage prefix for E5
    print(f"Target Chunk (index 20) for embedding:\n{target_chunk}\n")

    # Generate embeddings
    query_embedding = model.encode(query, normalize_embeddings=True)
    chunk_embedding = model.encode(target_chunk, normalize_embeddings=True)

    # Calculate cosine similarity
    cosine_similarity = util.cos_sim(query_embedding, chunk_embedding).item()
    print(f"Cosine Similarity between Query and Chunk 21: {cosine_similarity:.4f}")
else:
    print("Chunk 21 (index 20) not found in chunks.pkl.")