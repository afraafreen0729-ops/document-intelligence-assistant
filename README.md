# AI-Powered Document Intelligence and Research Assistant

## 1. Project Overview

The AI-Powered Document Intelligence and Research Assistant is an AI/ML-based web application that helps users analyse PDF documents, retrieve relevant information, generate AI-powered answers, and create document summaries.

The application supports multiple PDF uploads and provides source references, including document filenames and page numbers, to help users verify answers.

## 2. Problem Statement

Reading and analysing lengthy documents manually is time-consuming. Traditional keyword-based searches may also fail when the wording of a query differs from the wording in a document.

This project combines keyword retrieval, semantic search, and a language model to make document analysis easier and more efficient.

## 3. Objectives

* Extract and process text from PDF documents.
* Divide document text into smaller chunks.
* Retrieve relevant passages using keyword and semantic search.
* Generate answers grounded in retrieved document passages.
* Display document filenames and page references.
* Summarise lengthy PDF documents.
* Evaluate retrieval methods using a test dataset.

## 4. Features

* Multiple PDF upload.
* PDF text extraction using PyMuPDF.
* Text chunking with overlapping passages.
* TF-IDF keyword search.
* Semantic search using Sentence Transformers.
* FAISS vector similarity search.
* AI question answering using Qwen3 through Ollama.
* Source passages with document and page references.
* Document summarisation.
* Experimental similarity thresholds for filtering search results.
* Retrieval evaluation using Hit@1, Hit@2, and Mean Reciprocal Rank (MRR).

## 5. Technologies Used

| Technology            | Purpose                                      |
| --------------------- | -------------------------------------------- |
| Python                | Main programming language                    |
| Streamlit             | Web application interface                    |
| PyMuPDF               | PDF text extraction                          |
| Sentence Transformers | Text embeddings                              |
| all-MiniLM-L6-v2      | Embedding model                              |
| FAISS                 | Vector similarity search                     |
| Scikit-learn          | TF-IDF retrieval and similarity calculations |
| NumPy                 | Numerical operations                         |
| Ollama                | Local language model runtime                 |
| Qwen3                 | AI answer generation                         |

## 6. System Workflow

1. The user uploads one or more PDF documents.
2. The application extracts text from each PDF.
3. Extracted text is divided into overlapping chunks.
4. The user enters a question.
5. The selected retrieval method searches for relevant passages.
6. Retrieved passages are checked against the configured similarity threshold.
7. Relevant passages are supplied to the language model.
8. The language model generates an answer with source references.
9. Users can also generate a summary of a selected document.

## 7. Retrieval Evaluation

The project includes an evaluation script named `evaluation.py`.

The current evaluation compares TF-IDF against FAISS semantic search using:

* Hit@1
* Hit@2
* Mean Reciprocal Rank (MRR)

These metrics evaluate whether the correct source document appears near the top of the ranking.

Evaluation results depend on the test questions, relevant-source labels, and documents used. Results from the current small test dataset should be treated as preliminary rather than as proof of general performance.

## 8. Installation and Setup

### Prerequisites

* Python
* Visual Studio Code or another Python editor
* Ollama for local language model inference
* An internet connection for downloading pretrained models and dependencies

### Step 1: Open the Project Folder

Open the `document-intelligence-assistant` folder in VS Code.

### Step 2: Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Start Ollama

Make sure Ollama is installed and running. Verify the language model is available:

```powershell
ollama list
```

The application is configured to use:

```text
qwen3:4b-instruct
```

If the model is not installed, download the model tag configured in `app.py`:

```powershell
ollama pull qwen3:4b-instruct
```

### Step 5: Start the Application

```powershell
python -m streamlit run app.py --server.fileWatcherType none
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

### Step 6: Run the Evaluation

Place the required evaluation PDFs in the project directory and run:

```powershell
python evaluation.py
```

The evaluation script expects the filenames configured in `evaluation.py`.

## 9. Project Files

```text
document-intelligence-assistant/
├── .venv/
├── app.py
├── evaluation.py
├── requirements.txt
├── test_document.pdf
├── research_paper_1.pdf
└── research_paper_2.pdf
```

The PDF files listed above are local test documents and do not need to be included in a public repository.

## 10. Limitations

* The current PDF extraction pipeline relies on extractable text; scanned PDFs may require OCR.
* Retrieval quality depends on the question, document content, chunking, and retrieval method.
* Similarity thresholds are experimental and require additional evaluation.
* The language model may still produce incorrect answers, so source passages should be checked.
* Current evaluation results come from a small dataset and should not be generalised without further testing.
* Answer generation and summarisation speed depend on local hardware and model performance.

## 11. Future Enhancements

* Expand the labelled evaluation dataset.
* Improve retrieval quality and source citation verification.
* Measure response time and answer relevance.
* Improve the interface and user experience.
* Add OCR support for scanned PDFs.
* Explore reranking methods for better passage selection.
* Prepare comprehensive project documentation and a final evaluation report.

## 12. Expected Deliverables

* Functional web application.
* PDF processing and retrieval pipeline.
* AI question-answering and summarisation features.
* Retrieval evaluation script.
* Testing and evaluation report.
* GitHub repository and project documentation.
* Final project presentation.
