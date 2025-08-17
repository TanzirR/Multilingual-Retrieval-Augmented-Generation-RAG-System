import streamlit as st
import io
import sys
import time
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import create_qa_pipeline, answer_question, format_for_llm, get_system_prompt

# ---------------- Load Environment Variables ----------------
load_dotenv()

# ---------------- Streamlit Config ----------------
st.set_page_config(
    page_title="RAG Retrieval System (Bengali)",
    layout="wide",
    initial_sidebar_state="collapsed",  # Hide default sidebar
    page_icon="📚"
)

# ---------------- Custom CSS ----------------
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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📚 RAG Retrieval System</h1>
    <p>English & Bengali Text Retrieval with Hybrid Search & Re-ranking</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Session State Init ----------------
for key, default in {
    "query_history": [],
    "query_results": {},
    "selected_query": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- Cache Retriever ----------------
@st.cache_resource
def load_retriever():
    with st.spinner("🔄 Loading retrieval models..."):
        retriever = create_qa_pipeline()
        if retriever:
            return retriever
        st.error("❌ Failed to load retrieval models.")
        return None

# ---------------- Two-Column Layout (Query + Config) ----------------
run_clicked = False
client = None
show_debug = True
show_chunk_details = True

col_pad_left, col_config, col_query, col_pad_right = st.columns([0.6, 2, 3, 0.6], gap="large")

with col_config:
    st.header("Configuration")

    # API Key
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key:", value=default_api_key, type="password")

    if api_key:
        st.success("✅ API Key provided")
        client = OpenAI(api_key=api_key)
    else:
        st.warning("⚠️ Enter your OpenAI API key")

    # Model choice
    model_choice = st.selectbox("Model:", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    max_tokens = st.slider("Max Tokens:", 100, 2000, 1000, 100)

    # Retrieval Models
    retriever_instance = load_retriever()
    if not retriever_instance:
        st.stop()
    else:
        st.success("✅ Models Loaded")

    # Retrieval Parameters
    initial_k_for_reranking = st.slider("Initial Candidates (K1):", 10, 100, 40, 5)
    final_k_for_llm = st.slider("Final Chunks (K2):", 1, 15, 5, 1)
    semantic_weight = st.slider("Semantic Weight:", 0.0, 1.0, 0.85, 0.05)
    bm25_weight = 1.0 - semantic_weight
    st.info(f"BM25 Weight: {bm25_weight:.2f}")

    show_debug = st.checkbox("Show Debug Output", value=True)
    show_chunk_details = st.checkbox("Show Chunk Details", value=True)

    # ---------------- Recent Queries ----------------
    st.subheader("📝 Recent Queries")
    if st.session_state.query_history:
        for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            meta = st.session_state.query_results[hist_query]
            label = f"{i}. {hist_query[:40]} ({meta['timestamp']})"
            if st.button(label, key=f"history_{i}"):
                st.session_state.selected_query = hist_query
                st.rerun()
    else:
        st.info("No previous queries yet.")

with col_query:
    # ---------------- Query Input ----------------
    if st.session_state.selected_query:
        query = st.session_state.selected_query
        st.success(f"✅ Loaded results for: {query}")

        if st.button("✏️ New Query"):
            st.session_state.selected_query = None
            st.rerun()
    else:
        query = st.text_area("Query (in English or Bengali):", height=140)

    # Run button beside the query area
    run_clicked = st.button("🚀 Run Retrieval", type="primary", use_container_width=True)

    # ---------------- Results (under the button, same width as text area) ----------------
    if ("query" in locals()) and (query in st.session_state.query_results):
        result = st.session_state.query_results[query]
        st.success(f"✅ Retrieved {len(result['chunks'])} chunks in {result['retrieval_time']:.2f}s")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 AI Response", "📝 LLM Prompt", "📄 Chunks", "🔍 Debug", "📊 Analysis"])

        with tab1:
            if result['ai_response']:
                st.markdown(f"**Model Used:** {result['model_used']}")
                st.markdown(f"<div class='result-container'>{result['ai_response']}</div>", unsafe_allow_html=True)
            else:
                st.info("No AI response generated.")

        with tab2:
            st.code(result['llm_prompt'], language="markdown")

        with tab3:
            if show_chunk_details:
                for i, chunk in enumerate(result['chunks'], 1):
                    with st.expander(f"📄 Chunk {i} - Re-rank Score: {chunk.get('rerank_score', 0):.4f}"):
                        st.write(chunk.get('chunk_text', chunk.get('content', 'No content')))
            else:
                st.info("Chunk details hidden.")

        with tab4:
            if show_debug:
                st.text_area("Debug Output:", result['debug_output'], height=400)
            else:
                st.info("Debug output hidden.")

        with tab5:
            st.subheader("📊 Retrieval Analysis")
            scores_data = st.session_state.query_results.get(query, {}).get('scores_data', {})

            if scores_data:
                import pandas as pd

                df = pd.DataFrame(scores_data)
                df.set_index('Chunk', inplace=True)

                # Make sure all required score columns exist and are numeric
                for col in ['Re-rank Score', 'Hybrid Score', 'Semantic Score', 'BM25 Score']:
                    if col not in df.columns:
                        df[col] = 0.0
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

                # Plot the scores side-by-side for each chunk
                st.bar_chart(df[['Re-rank Score', 'Hybrid Score', 'Semantic Score', 'BM25 Score']])

                # Show average metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Re-rank Score", f"{df['Re-rank Score'].mean():.4f}")
                with col2:
                    st.metric("Avg Hybrid Score", f"{df['Hybrid Score'].mean():.4f}")
                with col3:
                    st.metric("Avg Semantic Score", f"{df['Semantic Score'].mean():.4f}")
                with col4:
                    st.metric("Avg BM25 Score", f"{df['BM25 Score'].mean():.4f}")
            else:
                st.info("No scores available.")

# ---------------- Retrieval Execution ----------------
if run_clicked:
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
        st.stop()

    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output

    start_time = time.time()
    try:
        final_chunks = answer_question(
            query, retriever_instance,
            initial_k_for_reranking=initial_k_for_reranking,
            final_k=final_k_for_llm,
            debug=True
        )
    except Exception as e:
        st.error(f"❌ Retrieval failed: {e}")
        sys.stdout = old_stdout
        st.stop()
    finally:
        sys.stdout = old_stdout

    retrieval_time = time.time() - start_time

    if not final_chunks:
        st.warning("⚠️ No chunks retrieved.")
        st.stop()

    # Build LLM Prompt
    llm_prompt = format_for_llm(query, final_chunks)
    system_prompt_for_llm = get_system_prompt()

    # Call OpenAI API
    ai_response = None
    if client:
        with st.spinner("🤖 Generating AI response..."):
            try:
                ai_response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_prompt_for_llm},
                        {"role": "user", "content": llm_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1
                ).choices[0].message.content
            except Exception as e:
                st.error(f"OpenAI API error: {e}")

    # Prepare scores
    scores_df = pd.DataFrame({
        'Chunk': [f"Chunk {i+1}" for i in range(len(final_chunks))],
        'Re-rank Score': [c.get('rerank_score', 0) for c in final_chunks],
        'Hybrid Score': [c.get('hybrid_score', 0) for c in final_chunks],
        'Semantic Score': [c.get('semantic_score', 0) for c in final_chunks],
        'BM25 Score': [c.get('bm25_score', 0) for c in final_chunks]
    }).set_index("Chunk")

    # Dict version for plotting and metrics
    scores_df_reset = scores_df.reset_index()
    scores_data = {
        'Chunk': scores_df_reset['Chunk'].tolist(),
        'Re-rank Score': scores_df_reset['Re-rank Score'].tolist(),
        'Hybrid Score': scores_df_reset['Hybrid Score'].tolist(),
        'Semantic Score': scores_df_reset['Semantic Score'].tolist(),
        'BM25 Score': scores_df_reset['BM25 Score'].tolist(),
    }

    # Store results in session
    st.session_state.query_results[query] = {
        'chunks': final_chunks,
        'llm_prompt': llm_prompt,
        'debug_output': captured_output.getvalue(),
        'retrieval_time': retrieval_time,
        'scores_df': scores_df,
        'scores_data': scores_data,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ai_response': ai_response,
        'model_used': model_choice
    }

    if query not in st.session_state.query_history:
        st.session_state.query_history.append(query)
        if len(st.session_state.query_history) > 10:
            oldest = st.session_state.query_history.pop(0)
            st.session_state.query_results.pop(oldest, None)

    st.session_state.selected_query = query
    st.rerun()

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>📚 RAG Retrieval System | Built by Tanzir Bin Razzaque | English & Bengali Text Processing</p>
    <p><small>Powered by multilingual-e5-base embedding model and CrossEncoder re-ranking</small></p>
</div>
""", unsafe_allow_html=True)
