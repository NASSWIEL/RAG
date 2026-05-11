# RAG


A Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering from PDF documents using state-of-the-art embeddings and Google's Gemini LLM.

## Overview

This RAG engine downloads PDF documents from URLs, processes them into semantic chunks, generates embeddings using HuggingFace's sentence transformers, and stores them in a vector index. When queried, it retrieves the most relevant document sections and uses Google Gemini to generate accurate, context-aware answers.

### Key Features

- **Automatic PDF Processing**: Downloads and extracts content from PDF URLs
- **Smart Caching**: Stores embeddings to disk — subsequent runs skip re-processing
- **Semantic Search**: Uses advanced HuggingFace embeddings for accurate retrieval
- **Gemini-Powered Responses**: Leverages Google's Gemini LLM for natural, accurate answers
- **Interactive CLI**: Ask questions in an interactive loop

### Architecture

The system consists of several components:

- **`rag_engine.py`**: Core RAG orchestrator — manages indexing, storage, and query pipeline
- **`pdf_loader.py`**: Downloads PDFs and extracts document content
- **`text_processor.py`**: Advanced text chunking and embedding configuration
- **`gemini_client.py`**: Initializes and configures the Gemini LLM
- **`main.py`**: Interactive CLI entry point

### How It Works

1. **Initialization**: On first run, the engine downloads the PDF, splits it into semantic chunks, generates embeddings, and persists them to `./storage/`
2. **Caching**: Subsequent runs detect existing embeddings (based on URL hash) and load them instantly
3. **Querying**: User questions trigger semantic search across document chunks; top-k results are passed to Gemini for response generation

## Installation

Le projet utilise `pyproject.toml` avec un runner `venv`. Installez les dépendances de production :

```bash
pip install -r requirements.txt
```

Pour installer les dépendances de développement (tests, linting, analyse statique) :

```bash
pip install --group dev -e .
```

Les outils de développement inclus sont : `bandit`, `gitlint-core`, `pyright`, `pytest`, `pytest-cov`, `ruff`.

## Configuration

Add your Google API key to `config.py`:

```python
GOOGLE_API_KEY = "your-api-key-here"
```

## Usage

Run the interactive query interface:

```bash
python main.py
```

The default PDF is set to `https://arxiv.org/pdf/2005.11401.pdf`. Modify `pdf_url` in `main.py` to index and query different documents.

### Example Session

```
Your question: What is the main contribution of this paper?
Answer: [Gemini-generated response based on retrieved context]

Your question: quit
```

## Requirements

- Python 3.12+
- Dépendances de production : `google-generativeai`, `llama-index`, `llama-index-embeddings-huggingface`, `llama-index-llms-gemini`, `llama-index-readers-file`, `sentence-transformers`, `torch`, `pypdf`
- Dépendances de développement : `bandit>=1.9.4`, `gitlint-core>=0.19.1`, `pyright>=1.1.409`, `pytest>=9.0.3`, `pytest-cov>=7.1.0`, `ruff>=0.15.12`


