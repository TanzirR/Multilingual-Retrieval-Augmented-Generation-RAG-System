import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from retrieve import (
    extract_keywords, 
    hybrid_rerank,
    format_for_manual_llm,
    create_copy_paste_output,
    get_system_prompt
)

# Page config
st.set_page_config(
    page_title="🔍 Bengali RAG System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources (cached)
@st.cache_resource
def load_resources():
    """Load FAISS index, chunks, and model"""
    try:
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        index = faiss.read_index("faiss_index.index")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return model, index, chunks, None
    except FileNotFoundError as e:
        return None, None, None, f"Files not found: {e}"
    except Exception as e:
        return None, None, None, f"Error loading resources: {e}"

# Main RAG pipeline
def rag_search_and_rank(query, top_k=7, semantic_weight=0.7):
    """Complete RAG pipeline with hybrid ranking"""
    model, index, chunks, error = load_resources()
    
    if error:
        return None, error
    
    try:
        # Step 1: Semantic search
        query_embedding = model.encode([f"query: {query}"], normalize_embeddings=True)
        scores, indices = index.search(query_embedding, top_k)
        
        # Step 2: Get retrieved chunks
        retrieved_chunks = [chunks[i] for i in indices[0]]
        raw_scores = scores[0]
        
        # Step 3: Hybrid reranking
        keyword_weight = 1.0 - semantic_weight
        reranked_results = hybrid_rerank(query, retrieved_chunks, raw_scores, semantic_weight, keyword_weight)
        
        return reranked_results, None
        
    except Exception as e:
        return None, f"Search error: {e}"

# Streamlit UI
def main():
    st.title("🔍 Complete Bengali RAG System")
    st.markdown("**Semantic Search + Keyword Reranking + Manual LLM Integration**")
    
    # Sidebar controls
    st.sidebar.markdown("### ⚙️ Settings")
    top_k = st.sidebar.slider("Retrieve Top K", 3, 15, 7)
    semantic_weight = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.7, 0.1)
    keyword_weight = 1.0 - semantic_weight
    
    st.sidebar.markdown(f"**Keyword Weight:** {keyword_weight:.1f}")
    
    # System status
    model, index, chunks, error = load_resources()
    if error:
        st.error(f"❌ System Error: {error}")
        st.info("Make sure you have:\n- `faiss_index.index`\n- `chunks.pkl`\n- sentence-transformers installed")
        return
    else:
        st.sidebar.success(f"✅ System Ready\n- {len(chunks)} chunks loaded\n- Model: multilingual-e5-base")
    
    # Main interface
    st.markdown("### 🔍 Ask Your Question")
    query = st.text_input(
        "Enter your question in Bengali or English:",
        placeholder="অনুপমের চরিত্র কেমন? / What is Anupam's character like?",
        help="You can ask questions in Bengali, English, or mixed language"
    )
    
    if query:
        # Perform RAG search
        with st.spinner("🔍 Searching and reranking..."):
            reranked_results, search_error = rag_search_and_rank(query, top_k, semantic_weight)
        
        if search_error:
            st.error(f"❌ Search failed: {search_error}")
            return
        
        # Show results
        st.success(f"✅ Found {len(reranked_results)} results")
        
        # Keywords analysis
        keywords = extract_keywords(query)
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🔍 **Keywords:** {', '.join(keywords)}")
        with col2:
            st.info(f"⚖️ **Weights:** Semantic={semantic_weight:.1f}, Keyword={keyword_weight:.1f}")
        
        # Top result highlight
        st.markdown("### 🏆 Best Answer")
        best = reranked_results[0]
        
        # Score breakdown
        score_col1, score_col2, score_col3 = st.columns(3)
        with score_col1:
            st.metric("Hybrid Score", f"{best['hybrid_score']:.3f}")
        with score_col2:
            st.metric("Semantic", f"{best['semantic_score']:.3f}")
        with score_col3:
            st.metric("Keyword", f"{best['keyword_score']:.3f}")
        
        st.markdown("**Keywords Found:** " + ", ".join(best['keywords_found']))
        st.code(best['chunk'], language="text")
        
        # LLM-ready output
        st.markdown("### 🤖 Copy to LLM (ChatGPT/Claude/Gemini)")
        
        top_chunks = [item['chunk'] for item in reranked_results[:5]]
        llm_input = format_for_manual_llm(query, top_chunks)
        
        # System prompt first
        with st.expander("📋 System Prompt (Copy this first)", expanded=False):
            system_prompt = get_system_prompt()
            st.code(system_prompt, language="text")
            if st.button("📋 Copy System Prompt"):
                st.write("✅ Copy the text above and paste it as your system prompt in ChatGPT/Claude")
        
        # Main LLM input
        st.text_area(
            "LLM Input (Copy this to your AI assistant):",
            value=llm_input,
            height=300,
            help="Copy this entire text and paste it into ChatGPT, Claude, or any LLM"
        )
        
        # Download option
        col1, col2 = st.columns([3, 1])
        with col2:
            st.download_button(
                "💾 Download",
                data=create_copy_paste_output(query, top_chunks),
                file_name=f"rag_input_{query[:20].replace(' ', '_')}.txt",
                mime="text/plain"
            )
        
        # Detailed results
        with st.expander("📊 All Results Detailed", expanded=False):
            for i, item in enumerate(reranked_results, 1):
                st.markdown(f"**Rank {i}**")
                
                # Metrics in columns
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("Hybrid", f"{item['hybrid_score']:.3f}")
                with met_col2:
                    st.metric("Semantic", f"{item['semantic_score']:.3f}")
                with met_col3:
                    st.metric("Keyword", f"{item['keyword_score']:.3f}")
                with met_col4:
                    st.metric("Keywords", len(item['keywords_found']))
                
                if item['keywords_found']:
                    st.markdown(f"**Keywords Found:** {', '.join(item['keywords_found'])}")
                
                st.code(item['chunk'], language="text")
                st.markdown("---")
        
        # Usage tips
        with st.expander("💡 How to Use with LLMs", expanded=False):
            st.markdown("""
            **Step-by-step guide:**
            
            1. **Copy the System Prompt** (click the section above)
            2. **Paste it** as your system prompt in ChatGPT/Claude
            3. **Copy the LLM Input** from the text area
            4. **Paste it** as your message to the AI
            5. **Get your answer** based on the retrieved context!
            
            **Supported LLMs:**
            - ✅ ChatGPT (GPT-4, GPT-3.5)
            - ✅ Claude (Anthropic)
            - ✅ Gemini (Google)
            - ✅ Any other conversational AI
            
            **Tips:**
            - Adjust semantic/keyword weights in sidebar
            - Try different top-k values for more/fewer results
            - Ask follow-up questions with the same context
            """)

# Example queries
def show_examples():
    st.markdown("### 💡 Example Questions")
    examples = [
        "অনুপমের চরিত্র কেমন?",
        "বিবাহ সম্পর্কে কী বলা হয়েছে?",
        "মামার ভূমিকা কী?",
        "What is the story about?",
        "অনুপমের বয়স কত?"
    ]
    
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(f"📝 {example[:15]}...", key=f"ex_{i}"):
                st.session_state.example_query = example

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🔍 Bengali RAG System | Built with Streamlit + FAISS + Sentence Transformers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_examples()
    show_footer()