
# RAG From Scratch with Ollama

> A beginner-friendly Retrieval-Augmented Generation (RAG) project built with Python, NumPy, and Ollama.

**Status:** Learning project / educational prototype

## Overview

Built a Retrieval-Augmented Generation (RAG) system from scratch using Python, Ollama, Qwen3, and Nomic embeddings. The application ingests Markdown documents, generates embeddings, retrieves relevant context using cosine similarity, and produces context-grounded answers through a local LLM. Demonstrates foundational knowledge of semantic search, embeddings, and LLM orchestration.

The included fictional Eclipse Bistro Employee Handbook is used as the knowledge base.

## Architecture

```text
Markdown documents
       |
       v
Document loading
       |
       v
Paragraph chunking
       |
       v
Embedding generation
(nomic-embed-text)
       |
       v
User question
       |
       v
Question embedding
       |
       v
Cosine similarity search
       |
       v
Top-K retrieved chunks
       |
       v
Context + question
       |
       v
Local LLM (Qwen3)
       |
       v
Generated answer
```

## Tech stack

- Python
- Ollama
- Qwen3 4B Instruct
- Nomic Embed Text
- NumPy
- Requests

## Requirements

- Python 3.10+
- Ollama installed and running
- The following Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
```

## Installation

Clone the repository:

```bash
git clone https://github.com/TudorIacob/rag-from-scratch-ollama.git
cd rag-from-scratch-ollama
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running, then start the application:

```bash
python rag.py
```

## Example questions

Using the Eclipse Bistro Employee Handbook:

- Who is authorized to approve guest refunds?
- What temperature should refrigerated food be kept at?
- What should an employee do if a guest reports a peanut allergy?
- What is the procedure for reporting sick before a shift?
- What is the emergency number in Romania?

The system retrieves relevant document chunks and uses them as context for its answer.

## Current limitations

This is an educational prototype and has several limitations:

- Documents are split by paragraphs rather than semantic or token-based chunking.
- Embeddings are generated again every time the application starts.
- Embeddings are stored only in memory.
- Retrieval uses basic cosine similarity with NumPy.
- The system does not provide persistent storage or document-level source citations.
- The LLM may still produce inaccurate answers despite the grounding prompt.
- The application currently supports Markdown files in the `docs` directory.



## Learning objectives

This project was built to understand the fundamentals of:

- Text embeddings
- Semantic search
- Vector similarity
- Retrieval-Augmented Generation
- Context construction
- Local LLM inference

## Disclaimer

This project is intended for educational and portfolio demonstration purposes only. It is not a production-ready RAG system.

The Eclipse Bistro handbook is fictional and is used solely as sample knowledge-base content.

## License

MIT
