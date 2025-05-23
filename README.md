# Project: Conversational RAG System with LangChain V2

This project implements an enhanced Retrieval Augmented Generation (RAG) system using LangChain. It supports conversational queries, configurable embedding providers (HuggingFace, Azure OpenAI), Google Generative AI (Gemini) for language modeling, ChromaDB for vector storage, and MongoDB for persistent chat message history.

## Features

-   **Conversational AI**: Remembers previous turns in a conversation for a given session ID to answer follow-up questions.
-   **Document Ingestion**: Supports loading text (`.txt`) and PDF (`.pdf`) files.
-   **Text Splitting**: Splits documents into manageable chunks.
-   **Configurable Embeddings**:
    -   HuggingFace sentence transformers (e.g., `all-MiniLM-L6-v2`).
    -   Azure OpenAI Embeddings.
-   **LLM Integration**: Uses Google Generative AI (Gemini Pro) for both rephrasing questions and generating final answers.
-   **Vector Store**: Utilizes ChromaDB for creating and persisting document embeddings.
-   **Persistent Chat History**: Optionally uses MongoDB to store and retrieve chat histories across sessions.
-   **RAG Chain with History**: Implements an advanced LangChain Expression Language (LCEL) chain that:
    1.  Rephrases follow-up questions to be standalone based on chat history.
    2.  Retrieves relevant document chunks from ChromaDB.
    3.  Generates answers using the LLM, considering both retrieved context and chat history.
-   **Example Usage**: Provides a script (`run_rag_examples.py`) to demonstrate RAG capabilities with different embedding providers on sample documents.
-   **Unit Tests**: Includes an updated suite of unit tests (`test_rag_core.py`) using extensive mocking for the core RAG system.

## Project Structure

```
.
├── rag_core.py                     # Core RAGSystem class, LLM/embedding/DB logic
├── run_rag_examples.py             # Example script to demonstrate RAG (V2)
├── test_rag_core.py                # Unit tests for rag_core.py (V2)
├── industrial_sample_data_v2/      # Directory created by run_rag_examples.py
│   ├── tech_manual_v2.txt
│   ├── hr_policy_v2.txt
│   └── simple_code_explanation_v2_py_as_txt.txt
├── rag_example_docs/               # Directory created by rag_core.py's __main__ block
│   ├── doc1.txt
│   └── doc2.txt
├── example_chroma_db/              # Default ChromaDB created by rag_core.py's __main__
├── ./chroma_db_examples_hf/        # ChromaDB for HuggingFace by run_rag_examples.py
├── ./chroma_db_examples_azure/     # ChromaDB for Azure by run_rag_examples.py
└── README.md                       # This file
```
*(Note: ChromaDB directories like `./example_chroma_db` are created during script execution if they don't exist.)*

## Setup and Installation

1.  **Python**: Ensure Python 3.8+ is installed.
2.  **Clone Repository**: (Assuming this project is in a Git repository)
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```
3.  **Install Dependencies**:
    The system relies on several key Python libraries. You can install them using pip:
    ```bash
    pip install langchain langchain-community langchain-huggingface langchain-google-genai langchain-openai pymongo chromadb sentence-transformers pypdf torch
    ```
    Breakdown of key new dependencies:
    -   `langchain-google-genai`: For Google Generative AI (Gemini) models.
    -   `langchain-openai`: For Azure OpenAI Embeddings (if used).
    -   `pymongo`: For MongoDB integration (chat history).
    -   `chromadb`: For the Chroma vector store.
    -   `sentence-transformers`: For HuggingFace embeddings.
    -   `pypdf`: For loading PDF documents.
    -   `torch`: A dependency for sentence-transformers.

## Environment Variables & API Keys

To use the full capabilities of this RAG system, you need to set up several environment variables. These are typically API keys or connection strings for the services used.

> **Important**: It's recommended to manage these secrets securely, for example, by using a `.env` file (with `python-dotenv` library, not included in this project's dependencies by default) or by setting them directly in your shell environment.

1.  **Google Generative AI (Required for LLM)**:
    -   `GOOGLE_API_KEY`: Your API key for Google AI Studio (Gemini).
    ```bash
    export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

2.  **Azure OpenAI Services (Optional, for Azure Embeddings)**:
    If you plan to use Azure OpenAI for embeddings, set the following:
    -   `AZURE_OPENAI_API_KEY`: Your API key for Azure OpenAI.
    -   `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI service endpoint URL (e.g., `https://your-service-name.openai.azure.com/`).
    -   `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`: The name of your embedding model deployment in Azure OpenAI.
    -   `AZURE_OPENAI_API_VERSION`: The API version for Azure OpenAI (e.g., `2023-05-15`).
    ```bash
    export AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_KEY"
    export AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
    export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="YOUR_AZURE_EMBEDDING_DEPLOYMENT"
    export AZURE_OPENAI_API_VERSION="YOUR_AZURE_API_VERSION"
    ```

3.  **MongoDB (Optional, for Persistent Chat History)**:
    If you want to use MongoDB to store chat history across sessions:
    -   `MONGODB_CONNECTION_STRING`: Your MongoDB connection URI (e.g., `mongodb://user:pass@host:port/database`).
    ```bash
    export MONGODB_CONNECTION_STRING="YOUR_MONGODB_CONNECTION_STRING"
    ```
    If not set, chat history will not be persisted between script runs or different sessions (the chain falls back to non-persistent behavior).

## Core Components

### `rag_core.py`

This is the heart of the RAG system.

-   **`RAGSystem` Class**:
    -   **`__init__(...)`**: The constructor is significantly enhanced.
        -   `embedding_provider`: "huggingface" or "azure_openai".
        -   `embedding_config`: Dictionary for specific embedding provider settings (e.g., model names, API keys if not from env).
        -   `llm_provider`: Currently defaults to "google" (Gemini).
        -   `llm_config`: Dictionary for LLM settings (e.g., API key if not from env, model name, temperature).
        -   `vector_store_config`: Config for ChromaDB (e.g., `persist_directory`, `collection_name`).
        -   `memory_config`: Config for MongoDB chat history (connection string, DB name, collection name).
        -   Initializes the chosen embedding model, Google's Gemini for RAG and question rephrasing, ChromaDB vector store, and text splitter.
    -   **`_initialize_embeddings(...)`**: Dynamically loads HuggingFace or Azure OpenAI embedding models.
    -   **`_initialize_llm(...)`**: Initializes Google's ChatGoogleGenerativeAI model, ensuring `convert_system_message_to_human=True` for Gemini compatibility.
    -   **`_get_standalone_question(...)`**: Uses an LLM call to rephrase a follow-up question based on chat history.
    -   **`_retrieve_and_format_context(...)`**: Fetches relevant documents from ChromaDB.
    -   **`_build_conversational_rag_chain()`**: Constructs the core conversational RAG chain using LCEL and `RunnableWithMessageHistory`. This chain integrates question rephrasing, context retrieval, the main RAG prompt (with `MessagesPlaceholder` for history), and the LLM. It configures `MongoDBChatMessageHistory` if MongoDB details are provided.
    -   **`ingest_documents(...)`**: Loads, splits, and ingests documents into the configured ChromaDB vector store, then persists the store.
    -   **`query(self, question: str, session_id: str)`**: Takes a user question and a `session_id`. It invokes the conversational RAG chain, which handles history retrieval/saving for that session.
-   The `if __name__ == "__main__":` block in `rag_core.py` provides a basic demonstration with default settings (HuggingFace embeddings, Google LLM, local ChromaDB, optional MongoDB).

### `run_rag_examples.py`

This script (V2) showcases the `RAGSystem` with more practical examples and configurations:
-   **Sample Data**: Creates sample text files in `industrial_sample_data_v2/`.
-   **Environment Checks**: Includes a `check_env_vars()` function to verify necessary API keys.
-   **Dual Embedding Demonstrations**:
    1.  Runs an example using **HuggingFace Embeddings** (with Google LLM), saving its ChromaDB to `./chroma_db_examples_hf/`.
    2.  If Azure environment variables are set, runs another example using **Azure OpenAI Embeddings** (with Google LLM), saving its ChromaDB to `./chroma_db_examples_azure/`.
-   **Conversational Queries**: For each configuration, it runs `run_demonstrations_v2` which:
    -   Uses unique `session_id`s for different topics (e.g., technical support, HR).
    -   Demonstrates follow-up questions to test conversational memory.
-   **Setup/Cleanup**: Manages creation and deletion of `industrial_sample_data_v2/`.

### `test_rag_core.py`

This file (V2) contains an extensive suite of unit tests for `rag_core.py`:
-   Uses Python's `unittest` framework with heavy use of `unittest.mock` (`patch`, `MagicMock`).
-   Mocks all external services (embedding models, LLMs, ChromaDB, MongoDB history).
-   Tests various initialization scenarios (HuggingFace, Azure, with/without MongoDB).
-   Tests document ingestion logic with mocked loaders and ChromaDB calls.
-   Tests conversational query processing, including:
    -   Queries with no history (rephrasing should be skipped).
    -   Queries with mocked chat history (rephrasing and context retrieval with rephrased question).
    -   Fallback behavior when MongoDB is not configured.
-   Manages test-specific directories for documents and ChromaDB persistence.

## How to Run

1.  **Install Dependencies**: See "Setup and Installation".
2.  **Set Environment Variables**: Crucially, set `GOOGLE_API_KEY`. For optional features, set Azure keys and/or `MONGODB_CONNECTION_STRING` as described in "Environment Variables & API Keys".

3.  **Run `rag_core.py` (Basic Demo)**:
    To see the self-contained demo in `rag_core.py` (uses HuggingFace embeddings by default):
    ```bash
    python rag_core.py
    ```
    This creates and uses `rag_example_docs/` and `example_chroma_db/`.

4.  **Run `run_rag_examples.py` (Advanced Demonstrations)**:
    This is the main script to showcase different configurations:
    ```bash
    python run_rag_examples.py
    ```
    This script will:
    -   Check for required environment variables.
    -   Create `industrial_sample_data_v2/`.
    -   Run examples with HuggingFace embeddings (ChromaDB in `./chroma_db_examples_hf/`).
    -   If Azure variables are set, run examples with Azure OpenAI embeddings (ChromaDB in `./chroma_db_examples_azure/`).
    -   Perform conversational queries for each setup.
    -   Clean up `industrial_sample_data_v2/`.
    > **Note**: The ChromaDB directories created by this script (`./chroma_db_examples_hf/`, `./chroma_db_examples_azure/`) are **not** automatically deleted by the script, allowing you to inspect them.

5.  **Run Unit Tests**:
    To execute the test suite (ensure development dependencies like `unittest.mock` are available):
    ```bash
    python test_rag_core.py
    ```
    This will run tests that mock external services. It creates and cleans up `test_sample_docs_v2/` and test-specific ChromaDB directories.

## Example Conversational Queries (from `run_rag_examples.py`)

The `run_rag_examples.py` script demonstrates conversations like:

**Session: Technical Support**
1.  User: "What should I do if I see error code E001?"
    *System retrieves context about E001, LLM answers.*
2.  User: "And what is the part number for the sensor module mentioned?"
    *System uses chat history to understand "mentioned" refers to E001 context, rephrases the question, retrieves context (possibly the same), LLM answers.*

**Session: HR Policy**
1.  User: "How many days of annual leave do full-time employees get?"
    *System retrieves context about leave, LLM answers.*
2.  User: "What about remote work?"
    *System uses chat history, rephrases, retrieves context about remote work, LLM answers.*

## Limitations and Future Enhancements

-   **LLM Choice**: Currently hardcoded to use Google's Gemini via `ChatGoogleGenerativeAI`. Could be made configurable to support other LLMs (e.g., OpenAI, local models).
-   **Error Handling**: Can be made more robust, especially around API calls and document processing.
-   **Scalability**: For very large datasets, ChromaDB performance on a single machine might be a bottleneck. Consider distributed vector databases.
-   **Advanced RAG**: Explore techniques like query transformation beyond rephrasing, hybrid search, or re-ranking of retrieved documents.
-   **Configuration Management**: Externalize more configurations (e.g., model names, prompts, ChromaDB settings) into a dedicated config file (e.g., YAML).
-   **Requirements File**: A formal `requirements.txt` or `pyproject.toml` should be maintained for production use.
-   **Security**: API keys in environment variables are standard for development but consider more secure secret management for production (e.g., HashiCorp Vault, cloud provider secret managers).
```
