import datetime
import os
import json # Added for SSE formatting
import asyncio # Added for stream_generator and client disconnect
from typing import Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Body, Request as FastAPIRequest # Added Request
from fastapi.responses import StreamingResponse # Added for streaming
from pydantic import BaseModel, Field
import uvicorn

# Attempt to import RAGSystem from rag_core
RAG_CORE_IMPORTED = False # Updated global flag name
try:
    from rag_core import RAGSystem, GOOGLE_GENAI_AVAILABLE, AZURE_OPENAI_AVAILABLE, MONGODB_AVAILABLE
    RAG_CORE_IMPORTED = True 
except ImportError:
    RAGSystem = None 
    GOOGLE_GENAI_AVAILABLE = False
    AZURE_OPENAI_AVAILABLE = False
    MONGODB_AVAILABLE = False
    print("WARNING: RAGSystem from rag_core.py could not be imported. API functionality will be limited.")


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query_text: str = Field(..., description="The query text to be processed by the RAG system.")
    session_id: str = Field(
        default_factory=lambda: f"api_session_{datetime.datetime.now(datetime.timezone.utc).timestamp()}_{os.urandom(4).hex()}",
        description="Unique session ID for maintaining conversational context. Auto-generated if not provided."
    )

# QueryResponse is no longer the primary response model for /query/ but kept for potential non-streaming use
class QueryResponse(BaseModel):
    session_id: str
    response_text: str
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

class StreamToken(BaseModel): # New model for streaming tokens
    token: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    rag_system_initialized: bool
    rag_core_imported: bool # Flag name updated in response model
    google_genai_available: bool
    azure_openai_available: bool
    mongodb_available: bool


# --- FastAPI App Instance ---
app = FastAPI(
    title="Conversational RAG API (Streaming)",
    description="An API for interacting with a Retrieval Augmented Generation system, "
                "supporting conversational context, configurable backends, and streaming responses.",
    version="1.1.0" # Version updated
)

# Global variable to hold the RAGSystem instance
rag_system_instance: Optional[RAGSystem] = None

# --- Startup Event Handler ---
@app.on_event("startup")
async def initialize_rag_system():
    global rag_system_instance
    if not RAG_CORE_IMPORTED or RAGSystem is None: # Use updated flag
        print("ERROR: RAGSystem class not imported. Cannot initialize RAG system for the API.")
        return

    print("--- Initializing RAGSystem for API (Streaming Version) ---")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    
    # Updated default paths/names for streaming API
    vector_store_persist_dir = os.getenv("VECTOR_STORE_PERSIST_DIRECTORY", "./api_chroma_db_stream")
    vector_store_collection_name = os.getenv("VECTOR_STORE_COLLECTION_NAME", "api_rag_collection_stream")
    mongo_db_name = os.getenv("MONGODB_DATABASE_NAME", "api_chat_history_stream")
    mongo_collection_name = os.getenv("MONGODB_COLLECTION_NAME", "api_chat_sessions_stream")

    if not google_api_key and GOOGLE_GENAI_AVAILABLE: print("WARNING: GOOGLE_API_KEY not set.")
    elif not GOOGLE_GENAI_AVAILABLE: print("WARNING: langchain-google-genai not installed.")

    embedding_config: Dict[str, Any] = {}
    if embedding_provider == "huggingface":
        embedding_config = {
            "model_name": os.getenv("HF_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            "model_kwargs": {'device': os.getenv("HF_EMBEDDING_DEVICE", "cpu")}
        }
        print(f"Using HuggingFace embeddings: {embedding_config['model_name']}")
    elif embedding_provider == "azure_openai":
        if not all([azure_openai_api_key, azure_openai_endpoint, azure_embedding_deployment, azure_api_version]):
            print("WARNING: Azure OpenAI environment variables not fully set.")
        elif not AZURE_OPENAI_AVAILABLE: print("WARNING: langchain-openai not installed.")
        embedding_config = {"azure_deployment": azure_embedding_deployment, "api_key": azure_openai_api_key,
                            "azure_endpoint": azure_openai_endpoint, "api_version": azure_api_version}
        print(f"Using Azure OpenAI embeddings: deployment '{azure_embedding_deployment}'")
    else:
        print(f"ERROR: Unsupported EMBEDDING_PROVIDER: {embedding_provider}. RAGSystem will not be initialized."); return

    llm_config: Dict[str, Any] = {"google_api_key": google_api_key,
                                  "model_name": os.getenv("LLM_MODEL_NAME", "gemini-pro"),
                                  "temperature": float(os.getenv("LLM_TEMPERATURE", 0.1))}
    vector_store_config: Dict[str, Any] = {"persist_directory": vector_store_persist_dir,
                                           "collection_name": vector_store_collection_name}
    print(f"Vector store (ChromaDB) persisting to: {vector_store_persist_dir}")

    memory_config: Dict[str, Any] = {}
    if mongodb_connection_string:
        if MONGODB_AVAILABLE:
            memory_config = {"mongodb_connection_string": mongodb_connection_string,
                             "mongodb_database_name": mongo_db_name,
                             "mongodb_collection_name": mongo_collection_name}
            print(f"MongoDB chat history enabled: db='{mongo_db_name}', collection='{mongo_collection_name}'")
        else: print("WARNING: MONGODB_CONNECTION_STRING set, but pymongo/langchain-mongodb not installed.")
    else: print("MongoDB chat history not configured.")
    
    try:
        rag_system_instance = RAGSystem(
            embedding_provider=embedding_provider, embedding_config=embedding_config,
            llm_provider="google", llm_config=llm_config,
            vector_store_config=vector_store_config, memory_config=memory_config
        )
        print("RAGSystem initialized successfully for the API (Streaming Version).")
    except Exception as e:
        print(f"ERROR: Failed to initialize RAGSystem: {e}"); rag_system_instance = None

# --- Streaming Generator ---
async def stream_generator(query: str, session_id: str, request_obj: FastAPIRequest) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events (SSE) for the RAG system's streamed response."""
    if rag_system_instance is None: # Should be caught before calling, but as a safeguard
        error_detail = "RAG System is not initialized or unavailable for streaming."
        error_event = StreamToken(token=error_detail).model_dump()
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
        return

    try:
        print(f"Starting stream for session_id='{session_id}', query='{query[:50]}...'")
        async for chunk in rag_system_instance.astream_query(question=query, session_id=session_id):
            if await request_obj.is_disconnected():
                print(f"Client disconnected for session_id='{session_id}'. Stopping stream.")
                # Optionally, perform cleanup or logging specific to disconnects
                break 
            
            sse_payload = StreamToken(token=chunk).model_dump()
            yield f"data: {json.dumps(sse_payload)}\n\n"
            await asyncio.sleep(0.01) # Small sleep to allow other tasks, if necessary

        # Optionally, send a closing event (though not strictly required by SSE spec for simple data streams)
        # yield "event: close\ndata: Stream ended\n\n"
        print(f"Stream finished for session_id='{session_id}'.")

    except asyncio.CancelledError:
        print(f"Stream cancelled for session_id='{session_id}' (client likely disconnected).")
    except Exception as e:
        print(f"ERROR during streaming for session_id='{session_id}': {e}")
        # import traceback; traceback.print_exc() # For server-side detailed logs
        error_detail = f"An error occurred while processing your streaming query: {str(e)}"
        error_event = StreamToken(token=error_detail).model_dump() # Use StreamToken for error payload consistency
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"


# --- API Endpoints ---
@app.post("/query/") # Removed response_model=QueryResponse
async def handle_stream_query(request_obj: FastAPIRequest, request_data: QueryRequest = Body(...)):
    """
    Processes a query using the RAG system and returns the response as a Server-Sent Event stream.
    Maintains conversation context using the `session_id`.
    """
    if rag_system_instance is None:
        # This error occurs before streaming starts, so a normal HTTPException is fine.
        raise HTTPException(status_code=503, detail="RAG System is not initialized or unavailable.")
    
    # Note: FastAPIRequest must typically be the first parameter if not using Depends.
    # The order here (request_obj: FastAPIRequest, request_data: QueryRequest) should work.
    # If issues, might need to adjust or use `Depends`. For now, assuming this works.
    return StreamingResponse(
        stream_generator(
            query=request_data.query_text, 
            session_id=request_data.session_id,
            request_obj=request_obj
        ), 
        media_type="text/event-stream"
    )

@app.get("/health/", response_model=HealthResponse)
async def health_check():
    is_rag_initialized = rag_system_instance is not None
    status = "ok" if is_rag_initialized else "unavailable"
    
    return HealthResponse(
        status=status,
        rag_system_initialized=is_rag_initialized,
        rag_core_imported=RAG_CORE_IMPORTED, # Use updated global flag
        google_genai_available=GOOGLE_GENAI_AVAILABLE,
        azure_openai_available=AZURE_OPENAI_AVAILABLE,
        mongodb_available=MONGODB_AVAILABLE
    )

# --- Main Block for Uvicorn ---
if __name__ == "__main__":
    print("\n--- Starting FastAPI RAG API (Streaming Version) ---")
    # Guidance on environment variables (remains largely the same, but emphasize API specifics)
    print("Key variables for RAGSystem (see rag_core.py for full list/defaults):")
    print("  - GOOGLE_API_KEY (required for LLM)")
    print("  - EMBEDDING_PROVIDER ('huggingface' or 'azure_openai')")
    print("  - Azure specific vars (if EMBEDDING_PROVIDER='azure_openai')")
    print("  - MONGODB_CONNECTION_STRING (optional, for persistent chat history)")
    print("  - VECTOR_STORE_PERSIST_DIRECTORY (API default: ./api_chroma_db_stream)")
    # ... (other relevant vars from startup_event)
    print("-------------------------------------\n")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```
