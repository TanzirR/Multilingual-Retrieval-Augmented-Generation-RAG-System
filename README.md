# 📚 Bengali & English RAG Retrieval System

This project implements a Retrieval Augmented Generation (RAG) pipeline designed to answer questions based on a provided PDF document, specifically handling both Bengali and English text. It features robust text extraction, intelligent chunking, hybrid retrieval (semantic and lexical search), and re-ranking to deliver highly relevant context for a Large Language Model (LLM).

## 🚀 Setup Guide

Follow these steps to set up and run the RAG retrieval system locally.

### Prerequisites

- **Python 3.8+**
- **Git**
- **Tesseract OCR Engine:**

  - **Windows**: Download and install from [Tesseract-OCR GitHub](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to select Bengali (ben) and English (eng) language data during installation. Add Tesseract to your system's PATH.
  - **macOS (Homebrew)**: `brew install tesseract` and `brew install tesseract-lang` (ensure ben and eng are installed).
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng`

- **Poppler (for pdf2image):**
  - **Windows**: Download Poppler for Windows from [here](https://poppler.freedesktop.org/). Extract it and add the bin folder to your system's PATH.
  - **macOS (Homebrew)**: `brew install poppler`
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install poppler-utils`

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
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
   (You will need to create a requirements.txt file containing all the libraries used, see "Used Tools" section for a list).

### Running the Pipeline

1. Ensure your input PDF file (e.g., `bangla-text.pdf`) is in the project root directory.

2. **Extract Text (OCR):**

   ```bash
   python ocr_pdf_dynamic.py
   ```

   This will generate `extracted_pages_data.json` and `segmented_document_data.json`.

3. **Chunk Text:**

   ```bash
   python chunks.py
   ```

   This will generate `structured_chunks.json`.

4. **Create Embeddings and FAISS Index:**

   ```bash
   python embedding.py
   ```

   This will generate `faiss_index.index` and `chunks_with_metadata.pkl`.

5. **Run the Streamlit UI (Retrieval Tester):**
   ```bash
   streamlit run app.py
   ```
   This will open the application in your web browser, allowing you to test queries.

## 🛠️ Used Tools, Libraries, and Packages

This project utilizes the following key tools and Python libraries:

- **pytesseract**: Python wrapper for Google's Tesseract-OCR Engine, used for text extraction from images (PDF pages).
- **pdf2image**: Converts PDF pages into PIL Image objects, enabling OCR.
- **Pillow (PIL)**: Python Imaging Library, used for image manipulation.
- **regex**: A more powerful regular expression module, used for text cleaning and pattern matching.
- **json**: For reading and writing structured data (page data, chunks with metadata).
- **numpy**: Fundamental package for numerical computation, used with FAISS.
- **pickle**: For serializing and deserializing Python objects (chunks and metadata).
- **faiss**: Facebook AI Similarity Search library, used for efficient similarity search on embeddings.
- **sentence-transformers**: For generating semantic embeddings (`intfloat/multilingual-e5-base`) and for re-ranking (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- **rank_bm25**: Implements the BM25 algorithm for lexical (keyword) search.
- **langchain**: Specifically `RecursiveCharacterTextSplitter` for intelligent text chunking.
- **streamlit**: For building the interactive web-based user interface.
- **io, sys, time, datetime**: Standard Python libraries for system interaction, timing, and date/time handling.

## 📝 Sample Queries and Outputs

To test the system, you can use the following sample queries in Bengali. The Streamlit UI will display the retrieved chunks, their scores, and the formatted LLM prompt.

### Sample Bengali Queries:

- আহসান হাবীবের জন্মস্থান কোথায়? (Where is Ahsan Habib's birthplace?)
- ভালোবাসা নামের অস্ত্র উত্তোলিত হলে কী হবে? (What will happen if the weapon named love is raised?)
- ট্রয়নগরী কিসের দৃষ্টান্ত? (What is Troy City an example of?)
- আহসান হাবীবকে কোন দুটি পুরস্কারে ভূষিত করা হয়? (Which two awards was Ahsan Habib honored with?)

### Sample Output Format (as seen in Streamlit's "LLM Prompt" tab):

```
Qusetion/প্রশ্ন: <Your Query Here>

Context/পরসঙ্গ:
CHUNK 1 (Page Range: <page_range>):
<Content of Retrieved Chunk 1>

CHUNK 2 (Page Range: <page_range>):
<Content of Retrieved Chunk 2>

... (up to Final_K chunks) ...

Give the answer. If you do not know the answer, say that you do not know the answer.
Your language should be based on the language used while questioning.
```

(Note: The actual LLM response is not generated by this application; you would copy the "Generated LLM Prompt" into an external LLM for an answer.)

## 📄 API Documentation

This project does not implement direct API calls to external Large Language Models (LLMs) for answer generation. The `app.py` provides a formatted prompt that can be manually copied and pasted into any LLM interface (e.g., Google Gemini, OpenAI Playground, local LLM interfaces like Ollama/LM Studio).

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

For a more rigorous quantitative evaluation of a RAG system, one would typically use metrics such as:

- **Context Relevance**: How relevant are the retrieved chunks to the query?
- **Faithfulness**: Does the generated answer only use information from the retrieved context?
- **Answer Relevance**: How relevant is the generated answer to the query?
- **Answer Correctness**: Is the generated answer factually correct based on the source?
- **Recall/Precision (of retrieval)**: How many relevant documents were retrieved out of all relevant documents, and how many retrieved documents were actually relevant?

## ❓ Answering Key Questions

### What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

We used **pytesseract** (a Python wrapper for Google's Tesseract OCR engine) in conjunction with **pdf2image** to extract text from the PDF. `pdf2image` converts each page of the PDF into an image, which `pytesseract` then processes to extract text. `unicodedata` and `re` (regex) were used for post-OCR cleaning.

**Why**: Tesseract is a powerful, open-source OCR engine with good support for multiple languages, including Bengali and English, making it a cost-effective and flexible choice for local processing.

**Challenges**: Yes, we faced significant formatting challenges:

- **Stylized Text/Logos**: Elements like "10 MINUTE SCHOOL" (which appears as "1SHUTE 5৫H00" in OCR) were misrecognized due to their stylized font and logo-like nature.
- **OCR Noise and Artifacts**: Scanned documents often introduce noise, leading to extraneous characters or fragmented words.
- **Layout Interpretation**: Tesseract, especially with certain Page Segmentation Modes (PSM), can struggle with complex layouts, causing text to be out of order or combined incorrectly. Initial attempts with psm 6 (single uniform block) yielded poor results, requiring experimentation with psm 3 (automatic page segmentation) and higher DPI settings to improve accuracy.
- **Mixed Languages**: While `lang='ben+eng'` was used, the quality of English extraction was still impacted by the overall OCR clarity and potentially the quality of the English text within the primarily Bengali document.

### What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

We employed a multi-faceted chunking strategy using `langchain.text_splitter.RecursiveCharacterTextSplitter` combined with custom structural segmentation:

1. **Structural Segmentation**: The `DocumentProcessor` first segments the entire PDF into logical blocks based on content patterns (e.g., "story," "vocabulary," "questions," "general"). This ensures that semantically distinct sections are kept separate.

2. **Recursive Character Text Splitting**: Within each logical segment, `RecursiveCharacterTextSplitter` is used.

   - **Separators Priority**: We defined a hierarchy of separators: `["।", ".", "!", "?", "\n\n", "\n", "—", ",", " ", ""]`. This prioritizes splitting by sentence-ending punctuation, then paragraph breaks, then line breaks, and finally individual characters as a last resort. This is crucial for maintaining semantic coherence by trying to keep sentences and paragraphs intact.
   - **Character Limit & Overlap**: `chunk_size` (e.g., 700 characters) and `chunk_overlap` (e.g., 200 characters) are used to control the size of the chunks and ensure context continuity between them.

3. **Post-processing**: A final step merges very small chunks (e.g., < 50 characters) with adjacent chunks to prevent fragmented context, and re-chunks excessively large chunks (e.g., > 1800 characters) to adhere to size limits.

**Why it works well for semantic retrieval:**
This strategy works well because it aims to create chunks that are:

- **Semantically Coherent**: By prioritizing sentence and paragraph boundaries, each chunk is more likely to represent a complete thought or idea.
- **Contextually Rich**: Overlap ensures that the beginning or end of a relevant passage is not lost if the query's key information spans a chunk boundary.
- **Manageable Size**: Chunks are small enough to be relevant to a specific query but large enough to provide sufficient context for an LLM without exceeding its context window.
- **Structurally Aware**: Segmenting by document type (story, questions, vocabulary) allows for specialized processing and metadata, improving the relevance of retrieved content for specific query types.

### What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

We used the **intfloat/multilingual-e5-base** model from sentence-transformers for generating embeddings.

**Why we chose it:**

- **Multilingual Capability**: This model is specifically designed to handle multiple languages, which is essential for our use case involving both Bengali and English text.
- **Strong Performance**: The E5 series models are known for their strong performance across various semantic similarity and retrieval tasks.
- **Efficiency**: The 'base' version offers a good balance between performance and computational efficiency, making it suitable for local deployment.

**How it captures the meaning of the text:**
The `intfloat/multilingual-e5-base` model is a transformer-based neural network. It captures the meaning of text by:

- **Contextual Embeddings**: It processes input text through multiple layers, understanding the relationships between words and phrases within their context.
- **Vector Representation**: It then outputs a fixed-size numerical vector (embedding) for each piece of text. Texts with similar meanings are mapped to vectors that are close to each other in this high-dimensional vector space.
- **"passage:" and "query:" Prefixes**: The E5 models are trained with specific prefixes ("passage: " for documents and "query: " for queries). These prefixes help the model differentiate between the type of input, optimizing the embeddings for retrieval tasks where a query needs to be compared against passages.

### How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

We use a **hybrid retrieval approach** to compare the query with stored chunks, combining two main methods:

#### 1. Semantic Search (Dense Retrieval):

- **Method**: The query is embedded using `intfloat/multilingual-e5-base`, and this query embedding is compared against the pre-computed embeddings of all chunks stored in a FAISS index.
- **Similarity Method**: We use Inner Product (IP) for similarity search in FAISS. Since the embeddings are normalized (unit vectors), Inner Product is mathematically equivalent to cosine similarity, which measures the angle between two vectors. A smaller angle (closer to 0) indicates higher similarity.
- **Storage Setup**: FAISS (Facebook AI Similarity Search) is chosen because it's an open-source library optimized for efficient similarity search and clustering of dense vectors. It allows for fast retrieval of the top-k most similar chunks from millions of embeddings.

#### 2. Lexical Search (Sparse Retrieval):

- **Method**: We use `BM25Okapi` (from rank_bm25 library). This method performs keyword-based matching, looking for exact term overlap and considering term frequency and inverse document frequency.
- **Why**: BM25 is excellent at capturing exact keyword matches, which semantic models might sometimes miss, especially for very specific or rare terms.

#### 3. Hybrid Combination and Re-ranking:

- **Initial Hybrid Score**: The results from both semantic and BM25 searches are combined using weighted scores (`semantic_weight=0.85`, `bm25_weight=0.15`). This creates a broader pool of initial candidates.
- **Cross-Encoder Re-ranking**: A `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` CrossEncoder model is then used to re-rank these initial candidates. Cross-encoders take the query and each candidate chunk as a pair, processing them together to produce a relevance score. This is computationally more intensive but provides a more fine-grained understanding of relevance than dual-encoder (semantic) models.

**Why this choice:**
This hybrid approach balances the strengths of both methods:

- **Semantic Search**: Captures conceptual similarity, even if exact keywords aren't present.
- **Lexical Search**: Ensures recall of documents with direct keyword matches.
- **Re-ranking**: Refines the initial retrieval, ensuring the most relevant chunks are prioritized for the LLM, leading to more accurate answers.
- **FAISS**: Provides the necessary speed and scalability for large document collections.

### How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

#### Ensuring Meaningful Comparison:

- **Embedding Quality**: The choice of `intfloat/multilingual-e5-base` is critical. Its training on diverse multilingual data helps it generate high-quality embeddings that capture semantic meaning effectively for both Bengali and English.
- **Hybrid Retrieval**: The combination of semantic and lexical search ensures that both the "meaning" (semantic) and "keywords" (lexical) of the query are considered. This reduces the chance of missing relevant chunks due to either pure semantic drift or lack of exact keyword matches.
- **Cross-Encoder Re-ranking**: This is the most important step for ensuring meaningful comparison. By processing the query and each candidate chunk together as a pair, the Cross-Encoder can deeply analyze their interaction and contextual relevance, providing a highly accurate relevance score that goes beyond simple vector distance.
- **Prompt Engineering for Embeddings**: Using the `passage:` and `query:` prefixes for the E5 model helps align the embeddings for optimal retrieval performance.
- **Chunking Strategy**: As discussed, our chunking strategy aims to create semantically coherent and contextually rich chunks, ensuring that each chunk represents a meaningful unit of information that can be effectively compared to a query.

#### What would happen if the query is vague or missing context?

- **Vague Query**: If the query is vague (e.g., "What about the poet?"), the retrieval system might return a broader set of chunks that touch upon various aspects of the poet's life or work. The re-ranker will still try to find the "most relevant" among these, but the overall relevance might be lower than for a specific query. The LLM would then have a more general context, potentially leading to a more general or less precise answer, or even an "আমি জানি না" if no truly specific answer can be inferred.

- **Missing Context (in a conversation)**: In a multi-turn conversation, if the current query relies on previous turns but the system doesn't explicitly pass that conversational history (which our current `app.py` doesn't inherently do without further modifications), the query might be treated as a standalone question. This could lead to:
  - **Irrelevant Retrieval**: The retriever might miss chunks that are relevant only when combined with the previous conversational context.
  - **Incomplete Answers**: The LLM, lacking the full conversational context, might provide an answer that doesn't fully address the user's intent or seems out of place given the dialogue history. This is why implementing conversational memory (query rewriting/expansion) is crucial for true long-short term memory.

### Do the results seem relevant? If not, what might improve them (e.g., better chunking, better embedding model, larger document)?

Based on the provided debug output for the query "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", the results did not seem relevant at the very top, as the chunk explicitly stating Kalyani's age ("পনেরো বছর") was not among the initial top 40 retrieved candidates. This indicates a potential area for improvement in the initial retrieval stage.

#### Potential Improvements:

**1. Chunking Strategy Refinement:**

- **Granularity for Specific Facts**: For very specific factual questions (like age, dates), the current chunking might sometimes split the direct answer from its most relevant surrounding context, or embed it within a larger chunk that isn't highly ranked for that specific detail. We could explore more aggressive splitting around specific factual patterns if such questions are common.
- **Metadata Leverage**: Ensure that metadata (like question numbers, segment types) is being fully leveraged during retrieval or re-ranking. For instance, if a query is identified as a "question about age," prioritize chunks from "story" or "question" segments that are known to contain such details.

**2. Embedding Model:**

- **Domain-Specific Fine-tuning**: While multilingual-e5-base is good, fine-tuning it on a dataset of Bengali educational texts or questions-answer pairs from this specific domain could significantly boost its relevance for domain-specific queries.
- **Alternative Multilingual Models**: Experiment with other state-of-the-art multilingual embedding models (e.g., newer versions of E5, or models from Cohere, Google's own embedding models if accessible) to see if they offer better performance for your specific data.

**3. Retrieval and Re-ranking Parameters:**

- **Adjust Hybrid Weights**: Experiment with `semantic_weight` and `bm25_weight` in `hybrid_retrieve_initial`. If exact factual recall is more important, slightly increasing `bm25_weight` might help, but this can also introduce noise.
- **Increase initial_k_for_reranking**: Retrieving a larger pool of initial candidates (e.g., 60-80 instead of 40) might increase the chance of the correct chunk being present, even if its initial hybrid score isn't top-tier. The Cross-Encoder can then pick it out.
- **Re-ranker Model**: While mmarco-mMiniLMv2-L12-H384-v1 is good, research if there are Cross-Encoder models specifically fine-tuned for Bengali or for factual question answering.

**4. Document Quality (Pre-processing):**

- **Improved OCR**: The OCR output still shows noise and misrecognition (e.g., "1SHUTE 5৫H00"). Improving the initial OCR accuracy (e.g., higher DPI, better PSM modes, or using cloud-based OCR services like Google Cloud Vision for cleaner text) would directly lead to better embeddings and more accurate retrieval. Garbled text cannot be meaningfully embedded.
- **Structured Data Extraction**: For sections like questions, instead of just OCRing raw text, using more advanced techniques to extract structured data (e.g., question number, options, correct answer) could enable more precise retrieval.

**5. Conversational Memory (for follow-up questions):**

- As discussed, for queries that rely on previous context (e.g., "তার বাবার নাম কী?" after asking about the poet), implementing query rewriting or history summarization using an LLM would significantly improve relevance.

By systematically addressing these areas, the relevance of the retrieved results can be substantially improved for a wider range
