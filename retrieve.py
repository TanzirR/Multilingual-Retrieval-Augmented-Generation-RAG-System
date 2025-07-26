import regex as re
import string
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder # Import CrossEncoder
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, chunks_file="chunks_with_metadata.pkl", index_file="faiss_index.index"):
        """Initialize hybrid retriever with BM25 and semantic search."""
        # Load chunks and FAISS index
        with open(chunks_file, "rb") as f:
            self.chunks_with_metadata = pickle.load(f) # Renamed to clearly indicate it holds metadata
        
        # Extract just the chunk content for operations that need only text
        self.chunks_content = [chunk['content'] for chunk in self.chunks_with_metadata]
        
        self.faiss_index = faiss.read_index(index_file)
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
        # Initialize CrossEncoder for re-ranking
        self.reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        
        # Initialize BM25 with tokenized chunks_content
        self.tokenized_chunks = [self.tokenize_for_bm25(chunk_text) for chunk_text in self.chunks_content]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        print(f"✅ Hybrid retriever initialized with {len(self.chunks_content)} chunks")
    
    def tokenize_for_bm25(self, text):
        """Tokenize text for BM25 (preserving Bengali and English words)."""
        tokens = re.findall(r'[\p{Bengali}\p{Latin}]+', text.lower())
        return tokens
    
    def get_semantic_scores(self, query, top_k=10): # top_k here is for FAISS initial search
        """Get semantic similarity scores using FAISS."""
        query_embedding = self.embedding_model.encode(
            f"query: {query}", 
            normalize_embeddings=True
        )
        
        D, I = self.faiss_index.search(
            np.array([query_embedding]).astype(np.float32), 
            top_k 
        )
        
        semantic_results = []
        for idx, chunk_idx in enumerate(I[0]):
            semantic_results.append((chunk_idx, float(D[0][idx])))
        
        return semantic_results
    
    def get_bm25_scores(self, query, top_k=10): # top_k here is for BM25 initial search
        """Get BM25 scores for exact keyword matching."""
        query_tokens = self.tokenize_for_bm25(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        num_scores = len(bm25_scores)
        top_indices = np.argsort(bm25_scores)[::-1][:min(top_k, num_scores)]
        
        bm25_results = []
        for chunk_idx in top_indices:
            bm25_results.append((chunk_idx, float(bm25_scores[chunk_idx])))
        
        return bm25_results
    
    def hybrid_retrieve_initial(self, query, initial_k_for_reranking, semantic_weight=0.85, bm25_weight=0.15):
        """
        Perform initial hybrid retrieval to get a larger pool of candidates for re-ranking.
        """
        semantic_results = self.get_semantic_scores(query, initial_k_for_reranking * 2) 
        bm25_results = self.get_bm25_scores(query, initial_k_for_reranking * 2) 
        
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            if max_bm25 > 0:
                bm25_results = [(idx, score / max_bm25) for idx, score in bm25_results]
        
        combined_scores = {}
        
        for chunk_idx, sem_score in semantic_results:
            if chunk_idx not in combined_scores:
                combined_scores[chunk_idx] = {'semantic': 0, 'bm25': 0}
            combined_scores[chunk_idx]['semantic'] = sem_score
        
        for chunk_idx, bm25_score in bm25_results:
            if chunk_idx not in combined_scores:
                combined_scores[chunk_idx] = {'semantic': 0, 'bm25': 0}
            combined_scores[chunk_idx]['bm25'] = bm25_score
        
        initial_candidates = []
        for chunk_idx, scores in combined_scores.items():
            hybrid_score = (semantic_weight * scores['semantic'] + 
                            bm25_weight * scores['bm25'])
            
            initial_candidates.append({
                'chunk_idx': chunk_idx,
                'chunk_text': self.chunks_content[chunk_idx], # Get content from self.chunks_content
                'metadata': self.chunks_with_metadata[chunk_idx]['metadata'], # Include full metadata
                'hybrid_score': hybrid_score, 
                'semantic_score': scores['semantic'],
                'bm25_score': scores['bm25']
            })
        
        initial_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return initial_candidates[:initial_k_for_reranking]
    
    def rerank_chunks(self, query, initial_candidates, final_k):
        """
        Re-rank a list of initial candidate chunks using a CrossEncoder model.
        """
        if not initial_candidates:
            return []

        # Use 'chunk_text' here
        rerank_pairs = [[query, candidate['chunk_text']] for candidate in initial_candidates]
        rerank_scores = self.reranker_model.predict(rerank_pairs)

        for i, score in enumerate(rerank_scores):
            initial_candidates[i]['rerank_score'] = float(score) 

        initial_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return initial_candidates[:final_k]

    def retrieve_and_rerank(self, query, initial_k_for_reranking=10, final_k=5):
        """
        Performs initial hybrid retrieval, then re-ranks the results.
        """
        print(f"\n--- Stage 1: Initial Hybrid Retrieval (Top {initial_k_for_reranking}) ---")
        initial_candidates = self.hybrid_retrieve_initial(query, initial_k_for_reranking)
        
        print(f"\n--- Stage 2: Cross-Encoder Re-ranking (Selecting Top {final_k}) ---")
        final_results = self.rerank_chunks(query, initial_candidates, final_k)

        return final_results

    def retrieve_with_debug(self, query, initial_k_for_reranking=15, final_k=5):
        """
        Retrieves and re-ranks chunks with detailed debug information.
        This method will handle all debug prints.
        """
        print(f"\n--- Stage 1: Initial Hybrid Retrieval (Top {initial_k_for_reranking}) ---")
        initial_candidates = self.hybrid_retrieve_initial(query, initial_k_for_reranking)

        print(f"\n--- Stage 2: Cross-Encoder Re-ranking (Selecting Top {final_k}) ---")

        # Collect chunks and their re-rank scores
        rerank_inputs = [(query, cand['chunk_text']) for cand in initial_candidates]
        rerank_scores = self.reranker_model.predict(rerank_inputs).tolist()

        # Add rerank_score to each candidate for sorting
        for i, cand in enumerate(initial_candidates):
            cand['rerank_score'] = rerank_scores[i]

        # --- TEMPORARY DEBUG PRINT: Print all initial candidates with their scores ---
        print("\n--- All Initial Candidates with Re-rank Scores (for debug) ---")
        debug_sorted_candidates = sorted(initial_candidates, key=lambda x: x['rerank_score'], reverse=True)
        for i, cand in enumerate(debug_sorted_candidates):
            print(f"   Rank {i+1} (Debug) | Chunk Index: {cand['chunk_idx']} | Re-rank Score: {cand['rerank_score']:.4f} | Initial Hybrid Score: {cand['hybrid_score']:.4f}")
            print(f"     Text Preview: {cand['chunk_text'][:100]}...")
            if 'metadata' in cand: # Display metadata for debug
                print(f"     Metadata: {cand['metadata']}")
        print("------------------------------------------------------------------")
        # --- END TEMPORARY DEBUG PRINT ---

        # Sort by re-rank score and select final_k
        final_results = sorted(initial_candidates, key=lambda x: x['rerank_score'], reverse=True)[:final_k]
        
        # --- Final Debug Print for the LLM Prompt ---
        print(f"\n🔍 Query: {query}")
        print(f"📊 Final {len(final_results)} Chunks after Re-ranking:")
        print("="*80)
        
        for i, result in enumerate(final_results, 1):
            print(f"\n📄 RANK {i}")
            print(f"   Re-rank Score: {result['rerank_score']:.4f}") 
            print(f"   Initial Hybrid Score: {result['hybrid_score']:.4f}")
            print(f"   Semantic:       {result['semantic_score']:.4f}")
            print(f"   BM25:           {result['bm25_score']:.4f}")
            print(f"   Chunk Index:    {result['chunk_idx']}") 
            print(f"   Text Preview: {result['chunk_text'][:100]}...") # Use 'chunk_text'
            if 'metadata' in result:
                print(f"   Metadata: {result['metadata']}") # Display full metadata
            
            # Show keyword matches
            query_tokens = set(self.tokenize_for_bm25(query))
            chunk_tokens = set(self.tokenize_for_bm25(result['chunk_text'])) # Use 'chunk_text'
            matches = query_tokens & chunk_tokens
            if matches:
                print(f"   Keywords Found: {', '.join(matches)}")
        
        return final_results # Return full results with all scores for debug purposes

# Legacy functions for backward compatibility (These remain outside the class, mostly unused now)
def normalize_text(text):
    """Non-destructive normalization that preserves Bengali characters"""
    puncts = string.punctuation + '।""…''–—'
    table = str.maketrans('', '', puncts)
    text = text.translate(table)
    return re.sub(r'\s+', ' ', text.strip().lower())

def extract_keywords(text):
    """Improved keyword extraction for Bengali and English"""
    stopwords = {
        "কি", "কে", "কেন", "কিভাবে", "কোথায়", "কখন", "কী", "থেকে", "এবং", "হয়ে", 
        "যে", "এই", "সেই", "তার", "যার", "করে", "হওয়া", "আর", "ও", "বা",
        "when", "what", "why", "how", "who", "whom", "is", "the", "a", "an", 
        "in", "on", "for", "of", "to", "at", "by", "with", "from", "and", "or"
    }
    
    words = re.findall(r'[\p{Bengali}\p{Latin}]{2,}', text.lower())
    return [word for word in words if word not in stopwords]

def calculate_keyword_score(query, chunk):
    """Calculate keyword overlap score between query and chunk."""
    query_keywords = set(extract_keywords(query))
    chunk_keywords = set(extract_keywords(chunk))
    
    if not query_keywords:
        return 0.0
    
    matches = query_keywords & chunk_keywords
    score = len(matches) / len(query_keywords)
    return score

def hybrid_rerank(query, chunks, semantic_scores, semantic_weight=0.7, keyword_weight=0.3):
    """
    Rerank chunks using hybrid semantic + keyword scoring.
    (This function is likely deprecated now that HybridRetriever has its own re-ranking)
    """
    results = []
    
    for i, (chunk, sem_score) in enumerate(zip(chunks, semantic_scores)):
        keyword_score = calculate_keyword_score(query, chunk)
        
        hybrid_score = (semantic_weight * float(sem_score) + 
                        keyword_weight * keyword_score)
        
        query_keywords = set(extract_keywords(query))
        chunk_keywords = set(extract_keywords(chunk))
        keywords_found = list(query_keywords & chunk_keywords)
        
        results.append({
            'chunk': chunk,
            'hybrid_score': hybrid_score,
            'semantic_score': float(sem_score),
            'keyword_score': keyword_score,
            'keywords_found': keywords_found,
            'rank': i + 1
        })
    
    results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    for i, result in enumerate(results):
        result['rank'] = i + 1
    
    return results

def format_for_manual_llm(query, top_chunks):
    """Format query and chunks for manual LLM input."""
    context = ""
    for i, chunk_dict in enumerate(top_chunks, 1): # top_chunks now contain dictionaries
        context += f"CHUNK {i}:\n{chunk_dict['chunk_text']}\n\n" # Access 'chunk_text'
    
    prompt = f"""প্রশ্ন: {query}

প্রসঙ্গ:
{context}

উপরের প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর বাংলায় দিন। যদি প্রাসঙ্গিক তথ্য না পাওয়া যায়, তাহলে 'আমি জানি না' বলুন।"""

    return prompt

def create_copy_paste_output(query, chunks):
    """Create a complete copy-paste ready output for LLMs."""
    system_prompt = get_system_prompt()
    user_input = format_for_manual_llm(query, chunks)
    
    output = f"""SYSTEM PROMPT:
{system_prompt}

USER INPUT:
{user_input}
"""
    return output

def get_system_prompt():
    """Get the system prompt for LLM."""
    return """You are a helpful AI assistant that answers questions based on provided Bengali text context. 

Instructions:
1. Answer questions using ONLY the information provided in the context chunks
2. Respond in Bengali (বাংলা)
3. If the answer is not found in the context, say "আমি জানি না" (I don't know)
4. Be accurate and specific
5. Quote relevant parts from the context when helpful
6. Keep answers concise but complete"""

# Updated main functions using the new HybridRetriever
def create_qa_pipeline():
    """Create the complete QA pipeline with hybrid retrieval."""
    try:
        retriever = HybridRetriever()
        return retriever
    except Exception as e:
        print(f"❌ Error initializing retriever: {e}")
        print("Make sure you have run embedding.py to create chunks_with_metadata.pkl and faiss_index.index")
        return None

def answer_question(question, retriever=None, initial_k_for_reranking=10, final_k=5, debug=True):
    """Answer a question using hybrid retrieval and re-ranking."""
    if retriever is None:
        retriever = create_qa_pipeline()
        if retriever is None:
            return "Error: Could not initialize retriever"
    
    if debug:
        # Call retrieve_with_debug directly for all debug output
        results = retriever.retrieve_with_debug(question, initial_k_for_reranking, final_k)
    else:
        # For non-debug, directly call the main re-ranking function
        results = retriever.retrieve_and_rerank(question, initial_k_for_reranking, final_k)
    
    # Return top chunks (which are dictionaries now) for LLM
    # The format_for_llm function will extract 'chunk_text'
    return results

def format_for_llm(question, chunks): # chunks will now be a list of dictionaries
    """Format question and chunks for LLM input."""
    context = "\n\n".join([f"CHUNK {i+1} (Page Range: {chunk['metadata'].get('page_range', 'N/A')}):\n{chunk['chunk_text']}" for i, chunk in enumerate(chunks)])
    return f"""
প্রশ্ন: {question}

প্রসঙ্গ:
{context}

উত্তরটি বাংলায় দিন। যদি প্রাসঙ্গিক তথ্য না পাওয়া যায়, বলুন 'আমি জানি না'।
"""

if __name__ == "__main__":
    # Test the hybrid retriever
    print("🧪 Testing Hybrid Retriever")
    print("="*50)
    
    # Initialize retriever
    retriever = create_qa_pipeline()
    if retriever:
        # Test questions
        test_questions = [
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "অনুপমের চরিত্র কেমন?",
            "মামার ভূমিকা কী?",
            "কল্যাণী কেন বিয়েতে রাজি ছিল না?"
        ]
        
        INITIAL_K_FOR_RERANKING = 40 
        FINAL_K_FOR_LLM = 5 
        
        # Capture all output to save to file
        all_output = []
        all_output.append("🧪 Testing Hybrid Retriever")
        all_output.append("="*50)
        
        for question in test_questions:
            separator = f"\n" + "="*80
            print(separator)
            all_output.append(separator)
            
            # Capture debug output by redirecting stdout temporarily
            import io
            import sys
            
            # Create string buffer to capture print statements
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            # Run the retrieval with debug
            results = answer_question(question, retriever, 
                                     initial_k_for_reranking=INITIAL_K_FOR_RERANKING, 
                                     final_k=FINAL_K_FOR_LLM, 
                                     debug=True)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Get the captured debug output
            debug_output = captured_output.getvalue()
            print(debug_output)  # Still print to console
            all_output.append(debug_output)
            
            # Add LLM prompt section
            llm_section = f"\n📝 LLM PROMPT (with {len(results)} chunks):\n" + "-"*40
            print(llm_section)
            all_output.append(llm_section)
            
            # Pass the list of dictionaries (results) directly to format_for_llm
            prompt = format_for_llm(question, results) 
            print(prompt)
            all_output.append(prompt)
        
        # Save all output to file
        output_file = "retrieve_output.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(all_output))
            print(f"\n✅ All retrieval output saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Error saving output file: {e}")
    else:
        print("❌ Failed to initialize retriever")