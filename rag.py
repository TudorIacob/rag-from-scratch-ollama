
import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

DOCS_DIR = Path("docs")
TOP_K = 3


def embed(texts):
    """Generate embeddings for a list of text chunks."""
    response = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts,
        },
        timeout=120,
    )


    response.raise_for_status()

    embeddings = response.json()["embeddings"]
    return np.array(embeddings)


def ask_llm(prompt):
    """Send a prompt to the local Ollama chat model."""
    response = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


def load_documents():
    """Load Markdown documents and split them into paragraph chunks."""
    chunks = []

    for file in DOCS_DIR.glob("*.md"):
        text = file.read_text(encoding="utf-8")

        file_chunks = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError(
            f"No Markdown chunks found in '{DOCS_DIR}'. "
            "Add .md files to the docs folder."
        )

    return chunks


def retrieve(question, chunks, chunk_vectors, k=TOP_K):
    """Retrieve the k most semantically similar chunks."""
    question_vector = embed([question])[0]

    similarities = (
        chunk_vectors @ question_vector
        / (
            np.linalg.norm(chunk_vectors, axis=1)
            * np.linalg.norm(question_vector)
        )
    )

    top_indices = np.argsort(similarities)[::-1][:k]

    return [chunks[i] for i in top_indices]


def answer_question(question, chunks, chunk_vectors):
    """Retrieve context and generate an answer using the LLM."""
    retrieved_chunks = retrieve(
        question,
        chunks,
        chunk_vectors,
    )

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using only the context below.
If the answer is not contained in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question: {question}
"""

    return context, ask_llm(prompt)


def main():
    print("Loading documents...")

    chunks = load_documents()

    print(f"Loaded {len(chunks)} chunks.")

    print("Generating embeddings...")
    chunk_vectors = embed(chunks)

    print(f"Indexed {len(chunks)} chunks.")
    print(f"Embedding shape: {chunk_vectors.shape}")

    print("\nRAG system ready!")
    print("Ask a question, or press Enter to exit.\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            break

        try:
            context, answer = answer_question(
                question,
                chunks,
                chunk_vectors,
            )

            print("\nRetrieved context:")
            print(context)

            print("\nAnswer:")
            print(answer)
            print()

        except requests.RequestException as error:
            print(f"\nOllama request failed: {error}\n")


if __name__ == "__main__":
    main()
