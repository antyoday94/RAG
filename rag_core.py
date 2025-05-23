import os
import shutil
import asyncio # Added for astream_query example
from typing import List, Optional, Tuple, Dict, Any, AsyncGenerator

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory # Ensured import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Attempt to import optional dependencies
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    class ChatGoogleGenerativeAI: pass 

try:
    from langchain_openai import AzureOpenAIEmbeddings
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    class AzureOpenAIEmbeddings: pass

try:
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    class MongoDBChatMessageHistory: pass


# --- Helper Functions ---
def format_docs(docs: List[Document]) -> str:
    """Concatenates page_content of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- Prompt Templates ---
REPHRASE_QUESTION_SYSTEM_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up \
question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)

RAG_SYSTEM_PROMPT_STR = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}
"""
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=RAG_SYSTEM_PROMPT_STR),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{standalone_question}"),
    ]
)


class RAGSystem:
    def __init__(
        self,
        embedding_provider: str = "huggingface", 
        embedding_config: Optional[Dict[str, Any]] = None,
        llm_provider: str = "google", 
        llm_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        text_splitter_config: Optional[Dict[str, Any]] = None,
    ):
        if embedding_config is None: embedding_config = {}
        if llm_config is None: llm_config = {}
        if vector_store_config is None: vector_store_config = {"persist_directory": "./chroma_db_default"}
        if memory_config is None: memory_config = {}
        if text_splitter_config is None: text_splitter_config = {"chunk_size": 1000, "chunk_overlap": 200}

        self.embedding_function = self._initialize_embeddings(embedding_provider, embedding_config)
        self.llm_rag = self._initialize_llm(llm_provider, llm_config, for_rag=True)
        self.llm_rephrase = self._initialize_llm(llm_provider, llm_config, for_rag=False)

        self.text_splitter = RecursiveCharacterTextSplitter(**text_splitter_config)
        
        self.vector_store = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=vector_store_config.get("persist_directory", "./chroma_db_default"),
            collection_name=vector_store_config.get("collection_name", "langchain")
        )
        self.retriever = self.vector_store.as_retriever(
            search_type=vector_store_config.get("search_type", "similarity"),
            search_kwargs=vector_store_config.get("search_kwargs", {"k": 3})
        )

        self.memory_config = memory_config
        # The actual chain (history-enabled or fallback) is stored in self.conversational_rag_chain
        self.base_rag_processing_chain = self._build_base_rag_processing_chain()
        self.conversational_rag_chain = self._build_conversational_rag_chain()


    def _initialize_embeddings(self, provider: str, config: Dict[str, Any]):
        if provider == "huggingface":
            model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            model_kwargs = config.get("model_kwargs", {'device': 'cpu'})
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        elif provider == "azure_openai":
            if not AZURE_OPENAI_AVAILABLE:
                raise ImportError("AzureOpenAIEmbeddings is not available. Please install langchain-openai.")
            return AzureOpenAIEmbeddings(
                azure_deployment=config.get("azure_deployment", os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")),
                api_key=config.get("api_key", os.environ.get("AZURE_OPENAI_API_KEY")),
                azure_endpoint=config.get("azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT")),
                api_version=config.get("api_version", os.environ.get("AZURE_OPENAI_API_VERSION")),
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _initialize_llm(self, provider: str, config: Dict[str, Any], for_rag: bool = True):
        if provider == "google":
            if not GOOGLE_GENAI_AVAILABLE:
                raise ImportError("ChatGoogleGenerativeAI is not available. Please install langchain-google-genai.")
            google_api_key = config.get("google_api_key", os.environ.get("GOOGLE_API_KEY"))
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY must be provided either in llm_config or as an environment variable.")
            
            model_name = config.get("model_name", "gemini-pro")
            temperature = config.get("temperature", 0.7 if for_rag else 0.0)
            
            return ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=google_api_key,
                temperature=temperature,
                convert_system_message_to_human=True 
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _get_standalone_question(self, data: Dict[str, Any]) -> str:
        chat_history_messages = data.get("chat_history", [])
        formatted_chat_history = []
        for msg in chat_history_messages:
            if isinstance(msg, HumanMessage): formatted_chat_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage): formatted_chat_history.append(f"Assistant: {msg.content}")
        
        prompt_input = {"chat_history": "\n".join(formatted_chat_history), "question": data["question"]}
        rephrased_question_message = self.llm_rephrase.invoke(REPHRASE_QUESTION_SYSTEM_PROMPT.format_prompt(**prompt_input))
        return rephrased_question_message.content

    def _retrieve_and_format_context(self, standalone_question: str) -> str:
        docs = self.retriever.invoke(standalone_question)
        return format_docs(docs)

    def _build_base_rag_processing_chain(self):
        """Builds the core RAG logic without history management."""
        return RunnablePassthrough.assign(
            standalone_question=lambda x: self._get_standalone_question(x) if x.get("chat_history") else x["question"]
        ).assign(
            context=lambda x: self._retrieve_and_format_context(x["standalone_question"])
        ) | RAG_PROMPT_TEMPLATE | self.llm_rag | StrOutputParser()

    def _build_conversational_rag_chain(self):
        if MONGODB_AVAILABLE and self.memory_config.get("mongodb_connection_string"):
            return RunnableWithMessageHistory(
                self.base_rag_processing_chain, # Use the separately defined base chain
                lambda session_id: MongoDBChatMessageHistory(
                    session_id=session_id,
                    connection_string=self.memory_config["mongodb_connection_string"],
                    database_name=self.memory_config.get("mongodb_database_name", "chat_history_db"),
                    collection_name=self.memory_config.get("mongodb_collection_name", "chat_sessions"),
                ),
                input_messages_key="question",
                history_messages_key="chat_history",
                output_messages_key="answer" 
            )
        else:
            print("Warning: MongoDB not configured or unavailable. Chat history will not be persistent across queries/sessions.")
            return self.base_rag_processing_chain # Fallback to the base chain

    def ingest_documents(self, doc_paths: List[str], batch_size: int = 100):
        all_docs: List[Document] = []
        for doc_path in doc_paths:
            try:
                if doc_path.endswith(".txt"): loader = TextLoader(doc_path, encoding='utf-8')
                elif doc_path.endswith(".pdf"): loader = PyPDFLoader(doc_path)
                else: print(f"Unsupported file type: {doc_path}. Skipping."); continue
                all_docs.extend(loader.load())
            except Exception as e: print(f"Error loading document {doc_path}: {e}. Skipping."); continue
        
        if not all_docs: print("No documents successfully loaded. Aborting ingestion."); return

        texts = self.text_splitter.split_documents(all_docs)
        print(f"Split {len(all_docs)} documents into {len(texts)} text chunks.")

        for i in range(0, len(texts), batch_size):
            self.vector_store.add_documents(texts[i:i + batch_size])
            print(f"Ingested batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

        self.vector_store.persist()
        print("Document ingestion complete and vector store persisted.")
        
        self.retriever = self.vector_store.as_retriever(
            search_type=getattr(self.vector_store, '_search_type', "similarity"),
            search_kwargs=getattr(self.vector_store, '_search_kwargs', {"k": 3})
        )
        # Re-assign self.conversational_rag_chain as it depends on the retriever and memory config
        self.conversational_rag_chain = self._build_conversational_rag_chain()
        print("Retriever and conversational RAG chain have been updated.")

    def query(self, question: str, session_id: str = "default_session") -> Optional[str]:
        if not self.conversational_rag_chain:
            print("Error: Conversational RAG chain not initialized.")
            return "Error: RAG system not fully initialized."

        config: RunnableConfig = {"configurable": {"session_id": session_id}}
        
        try:
            if isinstance(self.conversational_rag_chain, RunnableWithMessageHistory):
                # History-enabled chain expects only the "question" (input_messages_key)
                response = self.conversational_rag_chain.invoke({"question": question}, config=config)
            else:
                # Fallback chain (base_rag_processing_chain) expects "question" and optional "chat_history"
                # For a single query without explicit history, pass empty chat_history.
                print("Querying with non-persistent history (fallback chain).")
                response = self.conversational_rag_chain.invoke(
                    {"question": question, "chat_history": []}, 
                    config=config # Config might be used by underlying components even if not for history
                )
            return response
        except Exception as e:
            print(f"Error during query invocation: {e}")
            return f"An error occurred while processing your query: {str(e)}"

    async def astream_query(self, question: str, session_id: str = "default_session") -> AsyncGenerator[str, None]:
        """
        Streams the response from the RAG system using the conversational chain with history.
        Yields chunks of the response as they are generated.
        """
        if not self.conversational_rag_chain:
            print("Error: Conversational RAG chain not initialized for streaming.")
            yield "Error: RAG system not fully initialized for streaming."
            return

        config: RunnableConfig = {"configurable": {"session_id": session_id}}
        full_response_chunks = []
        
        try:
            stream_input: Dict[str, Any]
            if isinstance(self.conversational_rag_chain, RunnableWithMessageHistory):
                # History-enabled chain expects only the "question" (input_messages_key)
                stream_input = {"question": question}
                print(f"Streaming with history for session: {session_id}")
            else:
                # Fallback chain (base_rag_processing_chain) expects "question" and optional "chat_history"
                # For a single query without explicit history, pass empty chat_history.
                stream_input = {"question": question, "chat_history": []}
                print("Streaming with non-persistent history (fallback chain).")

            async for chunk in self.conversational_rag_chain.astream(stream_input, config=config):
                yield chunk
                full_response_chunks.append(chunk)
            
            # For debugging or potential manual history saving if needed
            # full_response = "".join(full_response_chunks)
            # print(f"\n--- Full streamed response for session {session_id} (debug): {full_response} ---")

        except Exception as e:
            print(f"Error during streaming query invocation for session {session_id}: {e}")
            yield f"An error occurred while processing your streaming query: {str(e)}"
            # import traceback
            # traceback.print_exc() # For server-side detailed logs


if __name__ == "__main__":
    print("Starting RAG System Example...")

    # --- Configuration (same as before) ---
    EMBEDDING_PROVIDER = "huggingface" 
    embedding_conf = {}
    if EMBEDDING_PROVIDER == "azure_openai":
        if not all(os.getenv(var) for var in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]):
            print("Azure OpenAI environment variables not set for __main__ example. Exiting.")
            exit(1)
        embedding_conf = {"azure_deployment": os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]}

    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY environment variable not set for __main__ example. Exiting.")
        exit(1)
    
    llm_conf = {"google_api_key": os.environ["GOOGLE_API_KEY"], "model_name": "gemini-pro", "temperature": 0.7}
    vector_store_conf = {"persist_directory": "./example_chroma_db", "collection_name": "example_collection"}
    
    mongodb_conn_str = os.getenv("MONGODB_CONNECTION_STRING")
    memory_conf = {}
    if mongodb_conn_str and MONGODB_AVAILABLE:
        memory_conf = {
            "mongodb_connection_string": mongodb_conn_str,
            "mongodb_database_name": "rag_chat_history_db_main",
            "mongodb_collection_name": "example_sessions_main"
        }
        print(f"MongoDB configured for chat history in __main__ example.")
    else:
        print("MongoDB not configured for __main__ example. Chat history will not be persistent.")

    try:
        rag_system = RAGSystem(
            embedding_provider=EMBEDDING_PROVIDER, embedding_config=embedding_conf,
            llm_config=llm_conf, vector_store_config=vector_store_conf, memory_config=memory_conf
        )
        print("RAGSystem initialized successfully for __main__ example.")
    except Exception as e:
        print(f"Error initializing RAGSystem in __main__ example: {e}"); exit(1)

    sample_doc_dir = "rag_example_docs_main"
    if os.path.exists(sample_doc_dir): shutil.rmtree(sample_doc_dir)
    os.makedirs(sample_doc_dir, exist_ok=True)
    doc_content = {
        "doc1.txt": "The sky is blue during the day. At night, it is often dark and filled with stars.",
        "doc2.txt": "Apples are a type of fruit. They can be red, green, or yellow.",
    }
    doc_paths = []
    for filename, content in doc_content.items():
        filepath = os.path.join(sample_doc_dir, filename)
        with open(filepath, "w", encoding='utf-8') as f: f.write(content)
        doc_paths.append(filepath)
    
    print(f"\nIngesting documents for __main__ example from: {sample_doc_dir}")
    rag_system.ingest_documents(doc_paths)

    # --- Synchronous Query Example (existing) ---
    print("\n--- Starting Synchronous Conversational Queries (existing example) ---")
    session_sync = "user_session_sync_main"
    q1_sync = "What color is the sky?"
    print(f"\nSession: {session_sync}, Query: {q1_sync}")
    r1_sync = rag_system.query(q1_sync, session_id=session_sync)
    print(f"Response: {r1_sync}")

    q2_sync = "And what about apples?"
    print(f"\nSession: {session_sync}, Query (follow-up): {q2_sync}")
    r2_sync = rag_system.query(q2_sync, session_id=session_sync)
    print(f"Response: {r2_sync}")

    # --- Asynchronous Streaming Query Example ---
    async def main_test_streaming():
        print("\n--- Starting Asynchronous Streaming Queries ---")
        session_async = "user_session_async_main"

        q1_async = "What color is the sky?"
        print(f"\nSession: {session_async}, Streaming Query: {q1_async}")
        print("Streamed Response: ", end="", flush=True)
        async for chunk in rag_system.astream_query(q1_async, session_id=session_async):
            print(chunk, end="", flush=True)
        print("\n--- End of stream ---")

        q2_async = "And what about apples?" # Follow-up
        print(f"\nSession: {session_async}, Streaming Query (follow-up): {q2_async}")
        print("Streamed Response: ", end="", flush=True)
        async for chunk in rag_system.astream_query(q2_async, session_id=session_async):
            print(chunk, end="", flush=True)
        print("\n--- End of stream ---")
        
        # Example of a new topic in the same async session
        q3_async = "Are stars visible during the day?"
        print(f"\nSession: {session_async}, Streaming Query: {q3_async}")
        print("Streamed Response: ", end="", flush=True)
        async for chunk in rag_system.astream_query(q3_async, session_id=session_async):
            print(chunk, end="", flush=True)
        print("\n--- End of stream ---")


    if GOOGLE_GENAI_AVAILABLE: # Only run asyncio part if LLM is available
        asyncio.run(main_test_streaming())
    else:
        print("\nSkipping streaming example as Google GenAI is not available.")

    # --- Clean up ---
    print(f"\nCleaning up sample document directory: {sample_doc_dir}")
    shutil.rmtree(sample_doc_dir)
    print(f"ChromaDB data for __main__ example remains at: {vector_store_conf['persist_directory']}")
    print("\nMain example script finished.")
```
