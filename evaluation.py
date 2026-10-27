import os
import numpy as np
import pymupdf
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# PDF files in our project folder
PDF_FILES = [
    "test_document.pdf",
    "research_paper_1.pdf",
    "research_paper_2.pdf"
]

TEST_CASES = [
    {
        "question": "What are the project objectives?",
        "relevant_sources": ["test_document.pdf"]
    },
    {
        "question": "Which technologies are proposed for the project?",
        "relevant_sources": ["test_document.pdf"]
    },
    {
        "question": "What problem does retrieval augmented generation address?",
        "relevant_sources": ["research_paper_1.pdf"]
    },
    {
        "question": "How does RAG combine retrieval with generation?",
        "relevant_sources": ["research_paper_1.pdf"]
    },
    {
        "question": "What is the purpose of Sentence-BERT embeddings?",
        "relevant_sources": ["research_paper_2.pdf"]
    },
    {
        "question": "Why is Sentence-BERT useful for semantic similarity search?",
        "relevant_sources": ["research_paper_2.pdf"]
    },
    {
        "question": "What is the project workflow?",
        "relevant_sources": ["test_document.pdf"]
    },
    {
        "question": "What is the problem statement?",
        "relevant_sources": ["test_document.pdf"]
    }
]


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


def load_documents():
    chunks = []

    for filename in PDF_FILES:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"PDF not found: {filename}. "
                "Check its filename and project folder."
            )

        print(f"Reading {filename}...")

        with pymupdf.open(filename) as document:
            for page_number, page in enumerate(document, start=1):
                page_chunks = create_chunks(page.get_text())

                for chunk_number, chunk in enumerate(
                    page_chunks, start=1
                ):
                    chunks.append({
                        "source": filename,
                        "page": page_number,
                        "chunk": chunk_number,
                        "text": chunk
                    })

    if not chunks:
        raise ValueError("No extractable text found in the PDFs.")

    return chunks


def rank_documents(scores, chunks):
    document_scores = {}

    for score, chunk in zip(scores, chunks):
        source = chunk["source"]

        if (
            source not in document_scores
            or score > document_scores[source]["score"]
        ):
            document_scores[source] = {
                "score": float(score),
                "page": chunk["page"]
            }

    return sorted(
        document_scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True
    )


def calculate_metrics(ranked_documents, relevant_sources):
    ranked_sources = [
        source for source, details in ranked_documents
    ]

    relevant_sources = set(relevant_sources)

    first_rank = next(
        (
            rank
            for rank, source in enumerate(ranked_sources, start=1)
            if source in relevant_sources
        ),
        None
    )

    hit_at_1 = int(
        bool(ranked_sources)
        and ranked_sources[0] in relevant_sources
    )

    hit_at_2 = int(
        any(source in relevant_sources for source in ranked_sources[:2])
    )

    mrr = 1 / first_rank if first_rank else 0

    return hit_at_1, hit_at_2, mrr


def main():
    print("Loading PDFs...")
    chunks = load_documents()

    texts = [item["text"] for item in chunks]

    print(f"\nDocuments loaded: {len(PDF_FILES)}")
    print(f"Text chunks created: {len(chunks)}")

    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating FAISS index...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("Preparing TF-IDF...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    results = {
        "TF-IDF": [],
        "FAISS Semantic Search": []
    }

    for test in TEST_CASES:
        question = test["question"]
        relevant_sources = test["relevant_sources"]

        print(f"\nQuestion: {question}")

        # TF-IDF retrieval
        query_vector = vectorizer.transform([question])

        tfidf_scores = cosine_similarity(
            query_vector,
            tfidf_matrix
        ).flatten()

        tfidf_ranking = rank_documents(tfidf_scores, chunks)

        # FAISS semantic retrieval
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        distances, indices = index.search(
            query_embedding,
            len(chunks)
        )

        semantic_scores = np.zeros(len(chunks), dtype=np.float32)

        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:
                semantic_scores[idx] = score

        semantic_ranking = rank_documents(semantic_scores, chunks)

        for method, ranking in [
            ("TF-IDF", tfidf_ranking),
            ("FAISS Semantic Search", semantic_ranking)
        ]:
            metrics = calculate_metrics(
                ranking,
                relevant_sources
            )

            results[method].append(metrics)

            top_source, top_details = ranking[0]

            print(f"  Method: {method}")
            print(f"  Top document: {top_source}")
            print(f"  Best matching page: {top_details['page']}")
            print(f"  Hit@1: {metrics[0]}")
            print(f"  Hit@2: {metrics[1]}")
            print(f"  MRR: {metrics[2]:.3f}")

    print("\n===== FINAL AVERAGE RESULTS =====")

    for method, scores in results.items():
        averages = np.mean(scores, axis=0)

        print(f"\n{method}")
        print(f"Hit@1: {averages[0]:.3f}")
        print(f"Hit@2: {averages[1]:.3f}")
        print(f"MRR:   {averages[2]:.3f}")


if __name__ == "__main__":
    main()