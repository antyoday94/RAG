# Project: Conversational RAG System with LangChain V2

This project implements an enhanced Retrieval Augmented Generation (RAG) system using LangChain. It supports conversational queries, configurable embedding providers (HuggingFace, Azure OpenAI), Google Generative AI (Gemini) for language modeling, ChromaDB for vector storage, and MongoDB for persistent chat message history. It also provides a FastAPI interface for interaction and a simple ChatGPT-like web interface to consume the streaming API.

## Features

-   **Conversational AI**: Remembers previous turns in a conversation for a given session ID to answer follow-up questions.
-   **FastAPI Interface**: Exposes RAG functionality via a RESTful API with health checks and query endpoints.
    -   **Streaming Queries**: The `/query/` endpoint now streams responses token-by-token using Server-Sent Events (SSE).
-   **ChatGPT-like Web Interface**: A simple frontend (`index.html`, `style.css`, `script.js`) that consumes the streaming API.
-   **Document Ingestion**: Supports loading text (`.txt`) and PDF (`.pdf`) files.
-   **Text Splitting**: Splits documents into manageable chunks.
-   **Configurable Embeddings**: HuggingFace, Azure OpenAI.
-   **LLM Integration**: Uses Google Generative AI (Gemini Pro).
-   **Vector Store**: Utilizes ChromaDB.
-   **Persistent Chat History**: Optionally uses MongoDB.
-   **RAG Chain with History**: Implements an advanced LangChain Expression Language (LCEL) chain.
-   **Example Usage**: Provides scripts to demonstrate RAG capabilities directly.
-   **Unit Tests**: Includes updated suites of unit tests for core RAG, FastAPI, and basic frontend interactions.

## Project Structure

```
.
├── main.py                         # FastAPI application exposing the RAG system
├── rag_core.py                     # Core RAGSystem class, LLM/embedding/DB logic
├── run_rag_examples.py             # Example script to demonstrate RAG (V2)
├── test_main_api.py                # Unit tests for the FastAPI application
├── test_rag_core.py                # Unit tests for rag_core.py
├── index.html                      # HTML file for the chat GUI
├── style.css                       # CSS file for the chat GUI
├── script.js                       # JavaScript file for the chat GUI logic
├── industrial_sample_data_v2/      # Directory created by run_rag_examples.py
│   # ... sample files
├── rag_example_docs_main/          # Directory created by rag_core.py's __main__
│   # ... sample files
├── example_chroma_db/              # Default ChromaDB by rag_core.py's __main__
├── ./api_chroma_db_stream/         # Default ChromaDB created by main.py (FastAPI)
├── ./chroma_db_examples_hf/        # ChromaDB by run_rag_examples.py (HF)
├── ./chroma_db_examples_azure/     # ChromaDB by run_rag_examples.py (Azure)
└── README.md                       # This file
```
*(Note: ChromaDB directories are created during script/API execution if they don't exist.)*

## Setup and Installation

(This section remains the same - ensure `requests` is listed for Python SSE client example if it wasn't before).
The current dependency list is:
```bash
pip install langchain langchain-community langchain-huggingface langchain-google-genai langchain-openai pymongo chromadb sentence-transformers pypdf torch fastapi uvicorn[standard] requests
```

## Environment Variables & API Keys

(This section remains largely the same. API default for `VECTOR_STORE_PERSIST_DIRECTORY` is `./api_chroma_db_stream`.)

## Core Components

(Descriptions for `rag_core.py`, `run_rag_examples.py`, `test_rag_core.py`, `main.py`, `test_main_api.py` remain largely the same but should reflect their latest functionalities.)

### Frontend GUI Files
-   **`index.html`**: The main HTML structure for the chat interface.
-   **`style.css`**: CSS styles for the chat interface, providing a clean and responsive layout.
-   **`script.js`**: Client-side JavaScript that handles:
    -   Sending user messages to the FastAPI backend's `/query/` endpoint.
    -   Receiving and processing Server-Sent Events (SSE) for streaming AI responses.
    -   Dynamically updating the chat message area with user and AI messages.
    -   Managing a `sessionId` in `localStorage` for conversation continuity.

## How to Run (Scripts)

(This section remains largely the same.)

## FastAPI API Interface

(This section remains largely the same, detailing the API endpoints, `curl` examples, and client-side consumption examples for Python and JavaScript `fetch`.)

## Running the ChatGPT-like Web Interface

This project includes a simple web-based graphical user interface (GUI) that interacts with the FastAPI backend to provide a chat experience.

### 1. Prerequisites

-   **FastAPI Backend Running**: The FastAPI server (`main.py`) must be running and configured correctly with the necessary environment variables (especially `GOOGLE_API_KEY`). Ensure it's accessible (typically at `http://localhost:8000`). Refer to the "FastAPI API Interface" section for instructions on running the backend.
-   **Web Browser**: A modern web browser that supports JavaScript and Server-Sent Events (e.g., Chrome, Firefox, Safari, Edge).

### 2. Serving the Frontend Files

The frontend consists of `index.html`, `style.css`, and `script.js`. These static files need to be served by an HTTP server. You cannot simply open `index.html` directly in your browser using a `file:///` URL due to browser security restrictions (CORS) when making `fetch` requests to the API.

A simple way to serve these files is using Python's built-in HTTP server:
1.  Navigate to the root directory of this project in your terminal.
2.  Run the following command:
    ```bash
    python -m http.server 8080
    ```
    (If you have Python 2, the command might be `python -m SimpleHTTPServer 8080`).
    This will start a basic HTTP server on port `8080` (or another port if `8080` is busy or you choose a different one).

    Any simple HTTP server can be used (e.g., `npx serve`, Live Server extension in VS Code).

### 3. Accessing the GUI

Once the frontend HTTP server is running (e.g., on port `8080`) and the FastAPI backend is running (on port `8000`), open your web browser and navigate to:
```
http://localhost:8080
```
(Replace `8080` if you used a different port for the Python HTTP server).

### 4. Using the Interface

-   **Chat Area**: The main area displays messages. User messages appear on the right, and AI responses (streamed token by token) appear on the left.
-   **Input Field**: Type your message in the text area at the bottom.
-   **Send**: Click the "Send" button or press Enter (without Shift) to send your message.
-   **Streaming Responses**: AI responses will appear token by token, simulating a typing effect.
-   **Session ID**: A unique `sessionId` is generated (or retrieved from `localStorage`) when you first load the page. This ID is sent with each query to maintain conversational context with the backend for the current browser session.

### 5. CORS (Cross-Origin Resource Sharing) Considerations

The frontend (served from, e.g., `http://localhost:8080`) makes requests to the backend API (running at `http://localhost:8000`). Since these are different "origins" (due to different ports), web browsers will enforce CORS policies.

**For this example setup to work smoothly, the FastAPI backend needs to be configured to allow requests from the frontend's origin.**

If you encounter CORS errors in your browser's console, you'll need to add CORS middleware to your FastAPI application (`main.py`). Here's a conceptual example of how to do this:

```python
# In main.py, add these imports:
from fastapi.middleware.cors import CORSMiddleware

# ... (rest of your FastAPI app setup) ...
app = FastAPI(...) # Your existing app definition

# Add CORS middleware
origins = [
    "http://localhost:8080", # Allow your frontend origin
    # You can add other origins if needed, e.g., "http://127.0.0.1:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# ... (rest of your main.py, including RAGSystem initialization, endpoints, etc.) ...
```
After adding this to `main.py`, restart the Uvicorn server. This will tell the browser that requests from `http://localhost:8080` are permitted.

## Example Conversational Queries (via API or Scripts or GUI)

(This section remains largely the same, explaining the concept of conversational queries.)

## Limitations and Future Enhancements

(This section remains largely the same.)
```
