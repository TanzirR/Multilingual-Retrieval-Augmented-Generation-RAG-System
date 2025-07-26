import streamlit as st
import io
import sys
from retrieve import create_qa_pipeline, answer_question, format_for_llm

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="RAG Retrieval Tester (Bengali)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 RAG Retrieval Tester (Bengali)")
st.markdown("Enter a query to test the hybrid retrieval and re-ranking process.")

# --- Initialize Retriever (Cached to load only once) ---
@st.cache_resource
def load_retriever():
    """Loads the HybridRetriever and caches it."""
    st.info("Loading retrieval models (this may take a moment)...")
    retriever = create_qa_pipeline()
    if retriever:
        st.success("Retrieval models loaded successfully!")
    else:
        st.error("Failed to load retrieval models. Check console for errors from retrieval.py.")
    return retriever

retriever_instance = load_retriever()

if retriever_instance is None:
    st.stop() # Stop the app if retriever failed to load

# --- User Input ---
st.subheader("Enter Your Query")
query = st.text_area("Query:", height=70)

col1, col2 = st.columns(2)
with col1:
    initial_k_for_reranking = st.slider(
        "Initial Candidates for Re-ranking (Initial_K):",
        min_value=10, max_value=100, value=40, step=5,
        help="Number of chunks retrieved by hybrid search before re-ranking."
    )
with col2:
    final_k_for_llm = st.slider(
        "Final Chunks for LLM (Final_K):",
        min_value=1, max_value=10, value=5, step=1,
        help="Number of top chunks passed to the LLM after re-ranking."
    )

if st.button("Run Retrieval"):
    if not query:
        st.warning("Please enter a query.")
    else:
        st.subheader("Retrieval Results & Debug Output")
        
        # --- Capture print statements from retrieve_with_debug ---
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Call answer_question with debug=True to get all print outputs
            final_retrieved_chunks_for_llm = answer_question(
                query, 
                retriever_instance, 
                initial_k_for_reranking=initial_k_for_reranking, 
                final_k=final_k_for_llm, 
                debug=True
            )
        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")
            final_retrieved_chunks_for_llm = [] # Ensure it's an empty list on error
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
        debug_output = captured_output.getvalue()
        
        st.text_area("Debug Output (from retrieve_with_debug):", debug_output, height=600)

        # --- Display Final LLM Prompt ---
        st.subheader("Generated LLM Prompt")
        if final_retrieved_chunks_for_llm:
            llm_prompt = format_for_llm(query, final_retrieved_chunks_for_llm)
            st.code(llm_prompt, language="markdown") # Use markdown for better readability
        else:
            st.warning("No chunks were retrieved for the LLM prompt.")