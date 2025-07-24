import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import base64

# Load the embedding model
@st.cache_resource
def load_model():
    model_name = "intfloat/multilingual-e5-base"
    return SentenceTransformer(model_name)

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    return faiss.read_index("faiss_index.index")

# Load chunks
@st.cache_resource
def load_chunks():
    with open("chunks.pkl", "rb") as f:
        return pickle.load(f)

# Preprocess text for E5
def preprocess(text):
    return f"query: {text}"

# Encode query
def embed_query(query, model):
    query_embedding = model.encode(preprocess(query), normalize_embeddings=True)
    return np.array([query_embedding])

# RAG formatting function
def format_rag_input(question, chunks):
    """Format the question and retrieved chunks for the RAG system"""
    
    formatted_input = f"""Question: {question}

Context:
"""
    
    for i, chunk in enumerate(chunks, 1):
        formatted_input += f"""=== CHUNK {i} ===
{chunk}

"""
    
    formatted_input += """Based on the context above, please answer the question. If the answer is not found in the context, say you don't know."""
    
    return formatted_input

# System prompt for RAG
def get_system_prompt():
    return """You are a helpful AI assistant that answers questions based on provided context from Bengali literature, specifically focusing on Rabindranath Tagore's works.

Instructions:
- Answer the question based ONLY on the context provided below
- If the answer is not found in the context, clearly state "I don't know" or "The information is not available in the provided context"
- Provide answers in both Bengali and English when possible
- Be precise and cite specific parts of the context when answering
- Do not make up information that is not explicitly stated in the context
- If the context contains relevant information but is incomplete, mention what you can determine and what is unclear"""

# Streamlit UI
st.set_page_config(page_title="🔎 Bengali RAG QA", layout="wide")

st.title("📚 Bengali-English Semantic Search & RAG System")
st.markdown("Search your knowledge base using natural language queries in **Bengali or English**, then format for LLM input.")

# Sidebar controls
st.sidebar.markdown("### ⚙️ Search Settings")
top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=5)
show_scores = st.sidebar.checkbox("Show Similarity Scores", value=True)
show_raw_chunks = st.sidebar.checkbox("Show Raw Chunks", value=True)

st.sidebar.markdown("### 📋 System Prompt")
if st.sidebar.button("Show System Prompt"):
    st.sidebar.text_area("Copy this system prompt:", value=get_system_prompt(), height=200)

# Load resources
try:
    model = load_model()
    index = load_faiss_index()
    chunks = load_chunks()
    st.success(f"✅ Loaded {index.ntotal} documents successfully!")
except Exception as e:
    st.error(f"❌ Error loading resources: {str(e)}")
    st.stop()

# Query box
query = st.text_input("🔍 Enter your query here:", placeholder="উপন্যাসে অনুপমের চরিত্র কেমন?")

if query:
    # Perform search
    query_vec = embed_query(query, model)
    scores, indices = index.search(query_vec, top_k)
    
    # Collect retrieved chunks
    retrieved_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        retrieved_chunks.append(chunks[idx])
    
    # Display search results
    if show_raw_chunks:
        st.markdown("### 🔁 Retrieved Chunks")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            chunk = chunks[idx]
            if show_scores:
                st.markdown(f"**Rank {rank}** — Similarity Score: `{score:.4f}`")
            else:
                st.markdown(f"**Chunk {rank}**")
            st.markdown(f"```text\n{chunk}\n```")
            st.markdown("---")
    
    # Format for RAG
    formatted_rag_input = format_rag_input(query, retrieved_chunks)
    
    # Create context for copying (original format)
    context = ""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        context += f"=== CHUNK {rank} ===\n{chunk}\n\n{'='*50}\n\n"

    # Copy to clipboard button (browser-based)
    def clipboard_button(text, label="📋 Copy to Clipboard"):
        b64 = base64.b64encode(text.encode()).decode()
        return f'''
            <button onclick="navigator.clipboard.writeText(atob('{b64}'))" style="
                background-color: #ff6b6b; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 4px; 
                cursor: pointer;
                font-size: 14px;
                margin: 5px;
            ">{label}</button>
        '''

    # RAG-formatted output section
    st.markdown("### 🤖 RAG-Formatted Input for LLM")
    st.markdown("**Copy this formatted input to your LLM (ChatGPT, Claude, etc.):**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_area("RAG-Formatted Input:", value=formatted_rag_input, height=400, key="rag_input")
    with col2:
        st.markdown(clipboard_button(formatted_rag_input, "📋 Copy RAG Input"), unsafe_allow_html=True)
        
        if st.button("📄 Download as Text"):
            st.download_button(
                label="💾 Download RAG Input",
                data=formatted_rag_input,
                file_name=f"rag_input_{query[:20].replace(' ', '_')}.txt",
                mime="text/plain"
            )

    # Alternative formats section
    st.markdown("### 📋 Alternative Formats")
    
    tab1, tab2, tab3 = st.tabs(["📝 Plain Context", "🔢 Numbered Context", "📊 JSON Format"])
    
    with tab1:
        st.text_area("Plain Context (for manual copying):", value=context, height=200)
        st.markdown(clipboard_button(context, "📋 Copy Plain Context"), unsafe_allow_html=True)
    
    with tab2:
        numbered_context = f"Question: {query}\n\nContext:\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            numbered_context += f"\n[{i}] {chunk}\n"
        st.text_area("Numbered Context:", value=numbered_context, height=200)
        st.markdown(clipboard_button(numbered_context, "📋 Copy Numbered"), unsafe_allow_html=True)
    
    with tab3:
        import json
        json_format = {
            "question": query,
            "context_chunks": retrieved_chunks,
            "system_prompt": get_system_prompt(),
            "metadata": {
                "top_k": top_k,
                "scores": scores[0].tolist()
            }
        }
        json_str = json.dumps(json_format, ensure_ascii=False, indent=2)
        st.text_area("JSON Format:", value=json_str, height=200)
        st.markdown(clipboard_button(json_str, "📋 Copy JSON"), unsafe_allow_html=True)

    # Quick stats
    st.markdown("### 📊 Search Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chunks Retrieved", len(retrieved_chunks))
    with col2:
        st.metric("Best Score", f"{scores[0][0]:.4f}")
    with col3:
        st.metric("Total Characters", sum(len(chunk) for chunk in retrieved_chunks))
    with col4:
        st.metric("Avg Chunk Size", f"{sum(len(chunk) for chunk in retrieved_chunks) // len(retrieved_chunks)}")

# Instructions section
st.markdown("---")
st.markdown("### 📖 How to Use")
st.markdown("""
1. **Enter your question** in Bengali or English
2. **Review retrieved chunks** to verify relevance
3. **Copy the RAG-formatted input** to your preferred LLM
4. **Use the system prompt** for better results
5. **Adjust Top K** in sidebar to get more/fewer chunks

**Example queries:**
- `অনুপমের চরিত্র কেমন?` (What is Anupam's character like?)
- `বিবাহ সম্পর্কে কী বলা হয়েছে?` (What is said about marriage?)
- `মামার ভূমিকা কী?` (What is the uncle's role?)
""")
