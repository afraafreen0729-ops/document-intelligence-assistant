import re
import requests
import streamlit as st
import pymupdf
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. APP CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("AI-Powered Document Intelligence Assistant")
## =====================================================
# PROFESSIONAL UI STYLING
# =====================================================

st.markdown("""
<style>
    /* Main application background and text */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #f5f7fb !important;
        color: #263548 !important;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #17365d !important;
        font-weight: 700 !important;
    }

    /* Body text, paragraphs and labels */
    p,
    label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"] p {
        color: #263548 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px dashed #8ba6c1 !important;
        border-radius: 12px;
        padding: 14px;
        color: #263548 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: none !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #263548 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #2457a7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e1e8f0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(20, 50, 80, 0.04);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #17365d !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2457a7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #173d78 !important;
        color: #ffffff !important;
    }

    /* Expandable source passages */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e1e8f0;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    [data-testid="stExpander"] summary {
        color: #17365d !important;
    }

    /* Text input */
    input,
    textarea {
        background-color: #ffffff !important;
        color: #263548 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Header */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. SEARCH SETTINGS
# =====================================================

SEMANTIC_THRESHOLD = 0.40
TFIDF_THRESHOLD = 0.20
TOP_K = 5


# =====================================================
# 3. LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# 4. TEXT CHUNKING
# =====================================================

def create_chunks(text, chunk_size=500, overlap=100):
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# =====================================================
# 5. OLLAMA API
# =====================================================

def call_ollama(prompt, num_predict=300):

    payload = {
        "model": "qwen3:4b-instruct",
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional document research assistant. "
                    "Answer using only the supplied document text. "
                    "Do not invent facts or citations. "
                    "Give clear, concise answers."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": num_predict,
            "num_ctx": 4096
        }
    }

    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    answer = response.json()["message"]["content"]

    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL
    ).strip()

    return answer


# =====================================================
# 6. DOCUMENT SUMMARISATION
# =====================================================

def summarize_document(document_chunks, progress_bar):

    # Larger batches reduce the number of AI requests.
    batch_size = 20

    batches = [
        document_chunks[i:i + batch_size]
        for i in range(0, len(document_chunks), batch_size)
    ]

    if not batches:
        return "No extractable text was found in this document."

    batch_summaries = []

    for batch_number, batch in enumerate(batches, start=1):

        passages = "\n\n".join(
            f"[Page {item['page']}]\n{item['text']}"
            for item in batch
        )

        prompt = (
            "Summarize the following passages in up to five concise "
            "bullet points. Preserve the main ideas and important "
            "findings. Use only the supplied information and include "
            "page references when useful.\n\n"
            f"{passages}"
        )

        summary = call_ollama(
            prompt,
            num_predict=120
        )

        batch_summaries.append(
            f"Section {batch_number}:\n{summary}"
        )

        progress_bar.progress(
            batch_number / (len(batches) + 1)
        )

    combined_summaries = "\n\n".join(batch_summaries)

    final_prompt = (
        "Create a clear, organised summary using the following "
        "section summaries. Include the main purpose, key concepts, "
        "methods, findings and limitations where available. "
        "Use headings and bullet points. Do not invent information. "
        "Retain any page references provided.\n\n"
        f"{combined_summaries}"
    )

    final_summary = call_ollama(
        final_prompt,
        num_predict=350
    )

    progress_bar.progress(1.0)

    return final_summary


# =====================================================
# 7. PDF UPLOAD AND PROCESSING
# =====================================================

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

all_chunks = []

if uploaded_files:

    try:
        extracted_chunks = []
        total_pages = 0

        for uploaded_file in uploaded_files:

            with pymupdf.open(
                stream=uploaded_file.getvalue(),
                filetype="pdf"
            ) as document:

                total_pages += len(document)

                for page_number, page in enumerate(
                    document,
                    start=1
                ):

                    page_chunks = create_chunks(page.get_text())

                    for chunk_number, chunk in enumerate(
                        page_chunks,
                        start=1
                    ):

                        extracted_chunks.append({
                            "source": uploaded_file.name,
                            "page": page_number,
                            "chunk": chunk_number,
                            "text": chunk
                        })

        all_chunks = extracted_chunks

        st.success("Documents processed successfully!")

        col1, col2, col3 = st.columns(3)

        col1.metric("PDF Documents", len(uploaded_files))
        col2.metric("Total Pages", total_pages)
        col3.metric("Text Chunks", len(all_chunks))

        with st.expander("View uploaded documents"):
            for uploaded_file in uploaded_files:
                st.write(f"📄 {uploaded_file.name}")

        if not all_chunks:
            st.warning(
                "No extractable text was found. "
                "Scanned PDFs may require OCR."
            )

    except Exception as error:
        all_chunks = []
        st.error(f"PDF processing failed: {error}")


# =====================================================
# 8. DOCUMENT SUMMARISATION INTERFACE
# =====================================================

if all_chunks:

    st.divider()
    st.subheader("📄 Document Summarisation")

    available_documents = sorted({
        item["source"] for item in all_chunks
    })

    selected_document = st.selectbox(
        "Choose a PDF to summarize:",
        available_documents
    )

    if st.button("Summarize Selected PDF"):

        selected_chunks = [
            item
            for item in all_chunks
            if item["source"] == selected_document
        ]

        progress_bar = st.progress(0)

        try:
            with st.spinner("Summarising your document..."):

                summary = summarize_document(
                    selected_chunks,
                    progress_bar
                )

            st.subheader("Document Summary")
            st.markdown(summary)

            st.download_button(
                "Download Summary",
                data=summary,
                file_name="document_summary.txt",
                mime="text/plain"
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to Ollama. "
                "Please make sure Ollama is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "Summarisation timed out. "
                "Try a shorter document."
            )

        except Exception as error:
            st.error(f"Summarisation failed: {error}")


# =====================================================
# 9. SEARCH AND QUESTION ANSWERING
# =====================================================

if all_chunks:

    st.divider()
    st.subheader("Ask Your Documents")

    question = st.text_input(
        "Enter your question:",
        placeholder="What is the purpose of RAG?"
    )

    method = st.radio(
        "Choose a retrieval method:",
        [
            "AI Semantic Search (FAISS)",
            "TF-IDF Keyword Search"
        ]
    )

    if st.button("Generate Answer", type="primary"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            texts = [
                item["text"]
                for item in all_chunks
            ]

            try:

                # -------------------------------------
                # FAISS SEMANTIC SEARCH
                # -------------------------------------

                with st.spinner("Searching documents..."):

                    if method == "AI Semantic Search (FAISS)":

                        model = load_model()

                        document_vectors = model.encode(
                            texts,
                            normalize_embeddings=True,
                            convert_to_numpy=True
                        ).astype(np.float32)

                        question_vector = model.encode(
                            [question],
                            normalize_embeddings=True,
                            convert_to_numpy=True
                        ).astype(np.float32)

                        index = faiss.IndexFlatIP(
                            document_vectors.shape[1]
                        )

                        index.add(document_vectors)

                        k = min(TOP_K, len(all_chunks))

                        scores, indices = index.search(
                            question_vector,
                            k
                        )

                        results = [
                            (int(index_id), float(score))
                            for index_id, score in zip(
                                indices[0],
                                scores[0]
                            )
                            if index_id >= 0
                        ]

                        results = [
                            (index_id, score)
                            for index_id, score in results
                            if score >= SEMANTIC_THRESHOLD
                        ]

                    # ---------------------------------
                    # TF-IDF KEYWORD SEARCH
                    # ---------------------------------

                    else:

                        vectorizer = TfidfVectorizer()

                        document_vectors = (
                            vectorizer.fit_transform(texts)
                        )

                        question_vector = (
                            vectorizer.transform([question])
                        )

                        scores = cosine_similarity(
                            question_vector,
                            document_vectors
                        ).flatten()

                        indices = np.argsort(scores)[::-1][
                            :min(TOP_K, len(all_chunks))
                        ]

                        results = [
                            (int(index_id), float(scores[index_id]))
                            for index_id in indices
                            if scores[index_id] > 0
                        ]

                        results = [
                            (index_id, score)
                            for index_id, score in results
                            if score >= TFIDF_THRESHOLD
                        ]

                # -------------------------------------
                # CHECK SEARCH RESULTS
                # -------------------------------------

                if not results:

                    st.warning(
                        "No sufficiently relevant passages were found. "
                        "Try rephrasing your question or uploading "
                        "a more relevant document."
                    )

                else:

                    retrieved_chunks = [
                        all_chunks[index_id]
                        for index_id, score in results
                    ]

                    context = "\n\n".join(
                        (
                            f"Document: {item['source']}\n"
                            f"Page: {item['page']}\n"
                            f"Passage: {item['text']}"
                        )
                        for item in retrieved_chunks
                    )

                    # ---------------------------------
                    # GENERATE AI ANSWER
                    # ---------------------------------

                    prompt = (
                        f"Question: {question}\n\n"
                        f"Document passages:\n{context}\n\n"
                        "Answer in 2-4 clear sentences using only "
                        "the supplied passages. Cite the source "
                        "filename and page number. If the passages "
                        "do not answer the question, say so clearly. "
                        "Do not invent facts or citations."
                    )

                    with st.spinner(
                        "Qwen3 is generating your answer..."
                    ):

                        answer = call_ollama(
                            prompt,
                            num_predict=256
                        )

                    st.subheader("AI-Generated Answer")
                    st.write(answer)

                    # ---------------------------------
                    # SHOW SOURCE PASSAGES
                    # ---------------------------------

                    st.subheader("Retrieved Source Passages")

                    for index_id, score in results:

                        item = all_chunks[index_id]

                        with st.expander(
                            f"{item['source']} — "
                            f"Page {item['page']} — "
                            f"Score: {score:.3f}"
                        ):

                            st.write(item["text"])

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to Ollama. "
                    "Please check that Ollama is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The AI took too long to respond. "
                    "Please try a shorter question."
                )

            except requests.exceptions.RequestException as error:
                st.error(f"Ollama request failed: {error}")

            except Exception as error:
                st.error(f"An error occurred: {error}")