# Personalized Research Assistant

## Overview

This project is a Streamlit-based web application that serves as a personalized research assistant. It allows users to upload documents (PDFs, plain text files, or images), index their content for retrieval-augmented generation (RAG), and ask questions about the uploaded content or general research topics. The assistant uses a local large language model (LLM) via Ollama for natural language processing, supports multimodal queries (e.g., analyzing images with questions), and includes optional web search integration without requiring API keys.

### Key Features
- **Document Upload and Processing**: Supports PDFs (text extraction), plain text files, and images (with optional OCR for indexing).
- **Vector Search (RAG)**: Uses FAISS for efficient similarity search on indexed documents, with optional persistence to disk.
- **Multimodal Support**: Handles image analysis using Ollama's multimodal models (e.g., Llava).
- **Conversational Memory**: Maintains chat history for context-aware responses.
- **Web Search**: Integrates free web search via DuckDuckGo (no API key required) for broader research.
- **Local LLM Integration**: Relies on Ollama for both LLM inference and embeddings, ensuring privacy and no cloud dependencies.
- **Configurable**: All settings (e.g., Ollama URL, models) are managed via environment variables in a `.env` file.

The project is designed to run entirely locally, assuming Ollama is installed and running. It's ideal for researchers, students, or anyone needing a private AI assistant for document-based queries.

<img width="1103" height="616" alt="Screenshot 2025-08-11 at 1 50 23 AM" src="https://github.com/user-attachments/assets/8e7e8cd6-7d4c-4727-b2a5-9c390c16ce76" />

## Prerequisites
- **Python**: 3.10+ (tested with 3.12).
- **Ollama**: Installed and running (download from [ollama.com](https://ollama.com)).
  - Pull required models: `ollama pull llava:7b` (for multimodal LLM) and `ollama pull nomic-embed-text` (for embeddings).
  - Start Ollama: `ollama serve`.
- **Tesseract OCR**: Required for image text extraction (optional for image indexing).
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
- **Virtual Environment**: Recommended for dependency isolation.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/zahid-ali-shah/research_assistant.git
   cd research_assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and customize if needed:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` to set Ollama URL, models, etc. (defaults should work if Ollama is local).
   - If persisting the FAISS index, set `FAISS_INDEX_PATH` to a directory (e.g., `./faiss_index`).

5. Run the application:
   ```bash
   streamlit run app.py
   ```
   Open your browser at [http://localhost:8501](http://localhost:8501).

## Usage
1. **Upload a Document**:
   - Upload a PDF, text file, or image.
   - PDFs and text files are automatically processed and indexed.
   - Images: Preview shown; click "Index image (use OCR)" to extract and index text via Tesseract.

2. **Ask Questions**:
   - Enter a query about your documents or a research topic.
   - Check "Include web search results" for external info via DuckDuckGo.
   - The assistant retrieves relevant document chunks (if indexed), analyzes images (if uploaded), and generates a response using Ollama.
   - Responses include sources from documents and web results.

3. **Conversational Flow**:
   - Follow-up questions use previous chat history for context.

4. **Persistence**:
   - If `FAISS_INDEX_PATH` is set, the vector index is saved/loaded for reuse across sessions.

## Configuration (.env)
See `.env.example` for details:
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`).
- `OLLAMA_MODEL`: LLM model (default: `llava:7b` for multimodal support).
- `EMBEDDING_MODEL`: Embedding model (default: `nomic-embed-text`).
- `FAISS_INDEX_PATH`: Optional path to persist FAISS index (e.g., `./faiss_index`).

Web search uses DuckDuckGo, which requires no API keys.

## Architecture
- `app.py`: Streamlit frontend for UI, file uploads, and query handling.
- `src/research_assistant.py`: Core logic for indexing, embeddings, vector store (FAISS), and LLM querying.
- `src/document_processor.py`: Parses PDFs, text files, and images (with OCR).
- `src/web_search_api.py`: Web search integration via DuckDuckGo.
- `src/config.py`: Loads settings from environment variables.
- `src/utils.py`: Utility functions (e.g., text chunking).
- `src/logger.py`: Centralized logging.

The app uses LangChain for orchestration but keeps dependencies minimal. Code is type-hinted and documented for clarity.

## Troubleshooting
- **Ollama Errors**: Ensure Ollama is running and models are pulled (`ollama list`). Check logs for connection issues.
- **OCR Failures**: Verify Tesseract is installed and in PATH (`tesseract --version`).
- **FAISS Persistence**: Ensure write permissions for `FAISS_INDEX_PATH`.
- **Web Search**: DuckDuckGo may rate-limit heavy usage; fallback to no results if fails.
- **Memory/Performance**: Large documents may require chunking adjustments or more RAM.

## Improvements and Limitations
- **Enhancements**:
  - Support additional file types (e.g., DOCX).
  - Add UI for managing indexed documents.
  - Include unit tests for core components.
- **Limitations**:
  - Requires local Ollama server.
  - Image analysis needs multimodal models like Llava.
  - No authentication for Streamlit app.

## Contributing
Fork the repo, make changes, and submit a pull request. Ensure code is clean, tested, and follows PEP 8.

## License
[MIT License](LICENSE)
