from retrieve import create_qa_pipeline, answer_question, format_for_llm

def examine_question_chunks(question, top_k=5, output_file="answer_output.txt"):
    """Examine and save top-k chunks for a specific question."""
    print(f"🔍 Examining question: {question}")
    print(f"📊 Retrieving top {top_k} chunks...")
    
    # Initialize retriever
    retriever = create_qa_pipeline()
    if retriever is None:
        print("❌ Failed to initialize retriever")
        return
    
    # Get results with debug information
    results = retriever.retrieve_with_debug(question, top_k)
    
    # Prepare output content
    output_content = f"""QUESTION ANALYSIS REPORT
========================

Question: {question}
Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Top-K: {top_k}

RETRIEVED CHUNKS:
================
"""
    
    for i, result in enumerate(results, 1):
        output_content += f"""
CHUNK {i}:
----------
Chunk Index: {result['chunk_idx']}
Hybrid Score: {result['hybrid_score']:.4f}
Semantic Score: {result['semantic_score']:.4f}
BM25 Score: {result['bm25_score']:.4f}

Text:
{result['chunk_text']}

Keywords Found:
"""
        # Show keyword matches
        query_tokens = set(retriever.tokenize_for_bm25(question))
        chunk_tokens = set(retriever.tokenize_for_bm25(result['chunk_text']))
        matches = query_tokens & chunk_tokens
        if matches:
            output_content += f"{', '.join(matches)}\n"
        else:
            output_content += "No exact keyword matches\n"
        
        output_content += "="*80 + "\n"
    
    # Add LLM prompt section
    top_chunks = [result['chunk_text'] for result in results]
    llm_prompt = format_for_llm(question, top_chunks)
    
    output_content += f"""

LLM PROMPT FOR MANUAL TESTING:
==============================
{llm_prompt}

ANALYSIS SUMMARY:
================
Total chunks retrieved: {len(results)}
Average hybrid score: {sum(r['hybrid_score'] for r in results) / len(results):.4f}
Average semantic score: {sum(r['semantic_score'] for r in results) / len(results):.4f}
Average BM25 score: {sum(r['bm25_score'] for r in results) / len(results):.4f}

Score Distribution:
"""
    
    for i, result in enumerate(results, 1):
        output_content += f"Rank {i}: Hybrid={result['hybrid_score']:.4f}, Semantic={result['semantic_score']:.4f}, BM25={result['bm25_score']:.4f}\n"
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"✅ Analysis saved to {output_file}")
        
        # Also print a summary to console
        print(f"\n📋 SUMMARY:")
        print(f"Question: {question}")
        print(f"Retrieved {len(results)} chunks")
        print(f"Best hybrid score: {results[0]['hybrid_score']:.4f}")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving to file: {e}")

if __name__ == "__main__":
    # Test with your specific question
    question = "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    examine_question_chunks(question, top_k=5, output_file="answer_output.txt")