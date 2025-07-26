import streamlit as st
import io
import sys
import json
import time
from datetime import datetime
from retrieve import create_qa_pipeline, answer_question, format_for_llm

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="RAG Retrieval System (Bengali)",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

# --- Custom CSS for styling ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .chunk-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stSidebar"] h2 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>📚 RAG Retrieval System</h1>
    <p>English & Bengali Text Retrieval with Hybrid Search & Re-ranking</p>
</div>
""", unsafe_allow_html=True)

# --- Initialize Retriever ---
@st.cache_resource
def load_retriever():
    with st.spinner("🔄 Loading retrieval models (this may take a moment)..."):
        retriever = create_qa_pipeline()
        if retriever:
            return retriever
        else:
            st.error("❌ Failed to load retrieval models. Check console for errors.")
            return None

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🤖 Model Status")
    retriever_instance = load_retriever()

    if retriever_instance:
        st.success("✅ Models Loaded")
        st.info("📊 Embedding Model: multilingual-e5-base")
        st.info("🔄 Re-ranker: stsb-xlm-r-multilingual")
    else:
        st.error("❌ Models Failed to Load")
        st.stop()

    st.subheader("🎛️ Retrieval Parameters")
    initial_k_for_reranking = st.slider("Initial Candidates (K1):", 10, 100, 40, 5)
    final_k_for_llm = st.slider("Final Chunks (K2):", 1, 15, 5, 1)

    st.subheader("🔧 Advanced Settings")
    semantic_weight = st.slider("Semantic Weight:", 0.0, 1.0, 0.85, 0.05)
    bm25_weight = 1.0 - semantic_weight
    st.info(f"BM25 Weight: {bm25_weight:.2f}")
    show_debug = st.checkbox("Show Debug Output", value=True)
    show_chunk_details = st.checkbox("Show Chunk Details", value=True)

# --- Session State Management ---
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'query_results' not in st.session_state:
    st.session_state.query_results = {}

# --- Recent Queries ---
if st.session_state.query_history:
    with st.sidebar:
        st.subheader("📝 Recent Queries")
        for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            if st.button(f"🔄 {hist_query[:30]}...", key=f"history_{i}"):
                # Set the selected query for loading
                st.session_state.selected_query = hist_query
                st.session_state.query_input = hist_query
                st.rerun()

# --- Main Content Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Enter Your Query")

    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

    query = st.text_area("Query (in Bengali):", value=st.session_state.query_input, height=100)

    if query != st.session_state.query_input:
        st.session_state.query_input = query

# Initialize variables
final_retrieved_chunks_for_llm = []
captured_output = io.StringIO()
retrieval_time = 0

# Check if we're loading a previous query
if 'selected_query' in st.session_state and st.session_state.selected_query in st.session_state.query_results:
    query = st.session_state.selected_query
    saved_results = st.session_state.query_results[query]
    final_retrieved_chunks_for_llm = saved_results['chunks']
    captured_output = io.StringIO(saved_results['debug_output'])
    retrieval_time = saved_results['retrieval_time']
    # Clear the selected query flag
    del st.session_state.selected_query

if st.button("🚀 Run Retrieval", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            status_text.text("🔄 Running hybrid retrieval...")
            progress_bar.progress(25)
            final_retrieved_chunks_for_llm = answer_question(
                query, retriever_instance,
                initial_k_for_reranking=initial_k_for_reranking,
                final_k=final_k_for_llm,
                debug=True
            )
            progress_bar.progress(75)
            status_text.text("✅ Retrieval completed!")
            retrieval_time = time.time() - start_time

        except Exception as e:
            st.error(f"❌ Error during retrieval: {e}")
            final_retrieved_chunks_for_llm = []
            retrieval_time = 0
        finally:
            sys.stdout = old_stdout
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

        # Save results after new retrieval
        if final_retrieved_chunks_for_llm and query.strip():
            llm_prompt = format_for_llm(query, final_retrieved_chunks_for_llm)
            scores_data = {
                'Chunk': [f"Chunk {i+1}" for i in range(len(final_retrieved_chunks_for_llm))],
                'Re-rank Score': [chunk.get('rerank_score', 0) for chunk in final_retrieved_chunks_for_llm],
                'Semantic Score': [chunk.get('semantic_score', 0) for chunk in final_retrieved_chunks_for_llm],
                'BM25 Score': [chunk.get('bm25_score', 0) for chunk in final_retrieved_chunks_for_llm]
            }
            
            st.session_state.query_results[query] = {
                'chunks': final_retrieved_chunks_for_llm,
                'llm_prompt': llm_prompt,
                'debug_output': captured_output.getvalue(),
                'retrieval_time': retrieval_time,
                'scores_data': scores_data,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

# Display results if we have any (either from new retrieval or loaded from history)
if final_retrieved_chunks_for_llm:
    st.success(f"✅ Retrieved {len(final_retrieved_chunks_for_llm)} chunks in {retrieval_time:.2f}s")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 LLM Prompt", "📄 Retrieved Chunks", "🔍 Debug Output", "📊 Analysis"])

    with tab1:
        st.subheader("📝 Generated LLM Prompt")
        # Use saved LLM prompt if available, otherwise generate it
        if query in st.session_state.query_results:
            llm_prompt = st.session_state.query_results[query]['llm_prompt']
        else:
            llm_prompt = format_for_llm(query, final_retrieved_chunks_for_llm)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.code(llm_prompt, language="markdown")
        with col2:
            if st.button("📋 Copy Prompt"):
                st.success("Copied to clipboard!")

    with tab2:
        st.subheader("📚 Retrieved Chunks Details")
        if show_chunk_details:
            for i, chunk in enumerate(final_retrieved_chunks_for_llm, 1):
                with st.expander(f"📄 Chunk {i} - Score: {chunk.get('rerank_score', 'N/A'):.4f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Re-rank Score", f"{chunk.get('rerank_score', 0):.4f}")
                        st.metric("Semantic Score", f"{chunk.get('semantic_score', 0):.4f}")
                    with col2:
                        st.metric("BM25 Score", f"{chunk.get('bm25_score', 0):.4f}")
                        st.metric("Chunk Index", chunk.get('chunk_idx', 'N/A'))
                    st.text_area("Content:", chunk.get('chunk_text', chunk.get('content', 'No content')), height=200, key=f"chunk_content_{i}_{hash(query)}")

    with tab3:
        if show_debug:
            st.subheader("🔧 Debug Output")
            debug_text = captured_output.getvalue() if hasattr(captured_output, 'getvalue') else str(captured_output)
            st.text_area("Debug Information:", debug_text, height=400)
        else:
            st.info("Debug output is disabled.")

    with tab4:
        st.subheader("📊 Retrieval Analysis")
        # Use saved scores data if available, otherwise generate it
        if query in st.session_state.query_results:
            scores_data = st.session_state.query_results[query]['scores_data']
        else:
            scores_data = {
                'Chunk': [f"Chunk {i+1}" for i in range(len(final_retrieved_chunks_for_llm))],
                'Re-rank Score': [chunk.get('rerank_score', 0) for chunk in final_retrieved_chunks_for_llm],
                'Semantic Score': [chunk.get('semantic_score', 0) for chunk in final_retrieved_chunks_for_llm],
                'BM25 Score': [chunk.get('bm25_score', 0) for chunk in final_retrieved_chunks_for_llm]
            }
        
        st.bar_chart(scores_data, x='Chunk', y=['Re-rank Score', 'Semantic Score', 'BM25 Score'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Re-rank Score", f"{sum(scores_data['Re-rank Score']) / len(scores_data['Re-rank Score']):.4f}")
        with col2:
            st.metric("Avg Semantic Score", f"{sum(scores_data['Semantic Score']) / len(scores_data['Semantic Score']):.4f}")
        with col3:
            st.metric("Avg BM25 Score", f"{sum(scores_data['BM25 Score']) / len(scores_data['BM25 Score']):.4f}")

    # Add export functionality
    if query in st.session_state.query_results:
        st.subheader("💾 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download JSON Results"):
                results_json = json.dumps(st.session_state.query_results[query], indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download",
                    data=results_json,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        with col2:
            if st.button("🗑️ Clear Query History"):
                st.session_state.query_history = []
                st.session_state.query_results = {}
                st.success("Query history cleared!")
                st.rerun()

# --- Session State Management ---
if query and len(final_retrieved_chunks_for_llm) > 0:
    if query not in st.session_state.query_history:
        st.session_state.query_history.append(query)
        if len(st.session_state.query_history) > 10:
            # Remove oldest query and its results
            oldest_query = st.session_state.query_history.pop(0)
            if oldest_query in st.session_state.query_results:
                del st.session_state.query_results[oldest_query]

# --- Footer ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>📚 RAG Retrieval System | Built by Tanzir Bin Razzaque | English & Bengali Text Processing</p>
    <p><small>Powered by multilingual-e5-base embedding model and CrossEncoder re-ranking</small></p>
</div>
""", unsafe_allow_html=True)