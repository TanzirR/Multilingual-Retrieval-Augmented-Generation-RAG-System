
# 📚 Bengali & English RAG Retrieval System

This project implements a Retrieval Augmented Generation (RAG) pipeline designed to answer questions based on a provided PDF document, specifically handling both Bengali and English text. It features robust text extraction, intelligent chunking, hybrid retrieval (semantic and lexical search), and re-ranking to deliver highly relevant context for a Large Language Model (LLM).

## 🚀 Setup Guide

Follow these steps to set up and run the RAG retrieval system locally.

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key

 
### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TanzirR/Multilingual-Retrieval-Augmented-Generation-RAG-System.git
   cd Multilingual-Retrieval-Augmented-Generation-RAG-System
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - **Windows**: `.\venv\Scripts\activate`
   - **macOS/Linux**: `source venv/bin/activate`

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **API Setup:**
   Create a file named .env in the root directory of the project. Add your OpenAI API key to this file.
   ```bash
   OPENAI_API_KEY="your_openai_api_key_here"
   ```
## Running the Pipeline

   Running main.py will run the entire RAG pipeline. As the argument, the directory and the name of the pdf is required. 
   ```bash
   python main.py ./data/document.pdf
   ```

## 🛠️ Used Tools, Libraries, and Packages

This project utilizes the following key tools and Python libraries:

- **easyocr**: A Python library for Optical Character Recognition (OCR) that supports over 80 languages including Bengali and English. Uses deep learning models for accurate text extraction.
- **pdf2image**: Converts PDF pages into PIL Image objects, enabling OCR.
- **Pillow (PIL)**: Python Imaging Library, used for image manipulation.
- **regex**: A more powerful regular expression module, used for text cleaning and pattern matching.
- **json**: For reading and writing structured data (page data, chunks with metadata).
- **numpy**: Fundamental package for numerical computation, used with FAISS and EasyOCR.
- **pickle**: For serializing and deserializing Python objects (chunks and metadata).
- **faiss**: Facebook AI Similarity Search library, used for efficient similarity search on embeddings.
- **sentence-transformers**: For generating semantic embeddings (`intfloat/multilingual-e5-base`) and for re-ranking (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- **rank_bm25**: Implements the BM25 algorithm for lexical (keyword) search.
- **langchain**: Specifically `RecursiveCharacterTextSplitter` for intelligent text chunking.
- **streamlit**: For building the interactive web-based user interface.
- **io, sys, time, datetime**: Standard Python libraries for system interaction, timing, and date/time handling.

## 📝 Sample Queries and Outputs

**Sample PDF**
<img width="989" height="533" alt="page_2" src="https://github.com/user-attachments/assets/bb8f39e3-8b08-4ee9-8fe2-ad0f5028a8f2" />

**Streamlit UI**
<img width="1751" height="830" alt="page_1" src="https://github.com/user-attachments/assets/ea45bdf0-b0a2-4809-9c18-4a4a8c7d7de8" />

**Retrieval Analysis**
<img width="1003" height="839" alt="page_3" src="https://github.com/user-attachments/assets/72619e76-320d-42aa-9229-e288a5126fcd" />

## 📊 Evaluation Matrix

A formal, automated evaluation matrix is not explicitly implemented within this codebase. However, the Streamlit UI provides several metrics for qualitative analysis and debugging of the retrieval performance:

- **Re-rank Score**: The final score after Cross-Encoder re-ranking, indicating the overall relevance.
- **Initial Hybrid Score**: The combined score from semantic and BM25 search before re-ranking.
- **Semantic Score**: The cosine similarity score from the FAISS vector search.
- **BM25 Score**: The lexical similarity score from the BM25 algorithm.
- **Chunk Index**: The original index of the chunk.
- **Text Preview & Full Content**: Allows visual inspection of the retrieved text.
- **Metadata**: Provides context like `page_range`, `type`, `segment_id`, etc.
- **Keywords Found**: Shows overlapping keywords between the query and the chunk.
- **Retrieval Time**: Measures the time taken for the retrieval process.

