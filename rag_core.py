import os
import shutil
from typing import List, Optional, Tuple, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Attempt to import optional dependencies
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    # Define a placeholder if not available so the class can be defined
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
        embedding_provider: str = "huggingface", # "huggingface" or "azure_openai"
        embedding_config: Optional[Dict[str, Any]] = None,
        llm_provider: str = "google", # Currently only "google" is implemented for LLM
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
        self.llm_rephrase = self._initialize_llm(llm_provider, llm_config, for_rag=False) # Potentially different config for rephrasing

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
        self.conversational_rag_chain = self._build_conversational_rag_chain()


    def _initialize_embeddings(self, provider: str, config: Dict[str, Any]):
        if provider == "huggingface":
            model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            model_kwargs = config.get("model_kwargs", {'device': 'cpu'})
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        elif provider == "azure_openai":
            if not AZURE_OPENAI_AVAILABLE:
                raise ImportError("AzureOpenAIEmbeddings is not available. Please install langchain-openai.")
            # AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            # should be set as environment variables or passed in config.
            # For simplicity, assuming they are in env vars or config directly maps to constructor args.
            return AzureOpenAIEmbeddings(
                azure_deployment=config.get("azure_deployment", os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")),
                api_key=config.get("api_key", os.environ.get("AZURE_OPENAI_API_KEY")),
                azure_endpoint=config.get("azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT")),
                api_version=config.get("api_version", os.environ.get("AZURE_OPENAI_API_VERSION")),
                # chunk_size=config.get("chunk_size", 16) # Max tokens for Azure OpenAI
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _initialize_llm(self, provider: str, config: Dict[str, Any], for_rag: bool = True):
        if provider == "google":
            if not GOOGLE_GENAI_AVAILABLE:
                raise ImportError("ChatGoogleGenerativeAI is not available. Please install langchain-google-genai.")
            # GOOGLE_API_KEY should be set as an environment variable or passed in config
            google_api_key = config.get("google_api_key", os.environ.get("GOOGLE_API_KEY"))
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY must be provided either in llm_config or as an environment variable.")
            
            model_name = config.get("model_name", "gemini-pro")
            temperature = config.get("temperature", 0.7 if for_rag else 0.0) # Different temp for rephrasing
            
            return ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=google_api_key,
                temperature=temperature,
                convert_system_message_to_human=True # Important for Gemini
            )
        # Add other LLM providers here (e.g., AzureOpenAIChat)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


    def _get_standalone_question(self, data: Dict[str, Any]) -> str:
        """Generates a standalone question from chat history and follow-up question."""
        # print(f"--- Debug: _get_standalone_question input data: {data} ---")
        chat_history_messages = data.get("chat_history", [])
        # Format chat history for the rephrasing prompt
        formatted_chat_history = []
        for msg in chat_history_messages:
            if isinstance(msg, HumanMessage):
                formatted_chat_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_chat_history.append(f"Assistant: {msg.content}")
        
        prompt_input = {
            "chat_history": "\n".join(formatted_chat_history),
            "question": data["question"]
        }
        # print(f"--- Debug: Rephrasing prompt input: {prompt_input} ---")
        rephrased_question_message = self.llm_rephrase.invoke(REPHRASE_QUESTION_SYSTEM_PROMPT.format_prompt(**prompt_input))
        # print(f"--- Debug: Rephrased question message: {rephrased_question_message} ---")
        return rephrased_question_message.content

    def _retrieve_and_format_context(self, standalone_question: str) -> str:
        """Retrieves documents and formats them into a context string."""
        # print(f"--- Debug: Retrieving context for: {standalone_question} ---")
        docs = self.retriever.invoke(standalone_question)
        # print(f"--- Debug: Retrieved docs: {len(docs)} ---")
        return format_docs(docs)

    def _build_conversational_rag_chain(self):
        # Define the core RAG logic using RunnablePassthrough and assignments
        # This part of the chain handles rephrasing, context retrieval, and the final RAG prompt
        base_rag_processing_chain = RunnablePassthrough.assign(
            # Conditionally rephrase question if chat_history is present
            standalone_question=lambda x: self._get_standalone_question(x) if x.get("chat_history") else x["question"]
        ).assign(
            # Retrieve context based on the (potentially rephrased) standalone question
            context=lambda x: self._retrieve_and_format_context(x["standalone_question"])
        ) | RAG_PROMPT_TEMPLATE | self.llm_rag | StrOutputParser()


        if MONGODB_AVAILABLE and self.memory_config.get("mongodb_connection_string"):
            chain_with_history = RunnableWithMessageHistory(
                base_rag_processing_chain,
                lambda session_id: MongoDBChatMessageHistory(
                    session_id=session_id,
                    connection_string=self.memory_config["mongodb_connection_string"],
                    database_name=self.memory_config.get("mongodb_database_name", "chat_history_db"),
                    collection_name=self.memory_config.get("mongodb_collection_name", "chat_sessions"),
                ),
                input_messages_key="question",      # User's query
                history_messages_key="chat_history", # Key for MessagesPlaceholder
                output_messages_key="answer"        # LLM's response (though StrOutputParser means it's a string)
            )
            return chain_with_history
        else:
            print("Warning: MongoDB not configured or unavailable. Chat history will not be persistent.")
            # Fallback to a chain without message history if MongoDB is not set up
            # This simplified chain won't have memory but will still use the rephrasing logic if chat_history is manually passed
            return base_rag_processing_chain


    def ingest_documents(self, doc_paths: List[str], batch_size: int = 100):
        """Loads, splits, and ingests documents into the vector store."""
        all_docs: List[Document] = []
        for doc_path in doc_paths:
            try:
                if doc_path.endswith(".txt"):
                    loader = TextLoader(doc_path, encoding='utf-8')
                elif doc_path.endswith(".pdf"):
                    loader = PyPDFLoader(doc_path)
                else:
                    print(f"Unsupported file type: {doc_path}. Skipping.")
                    continue
                loaded_docs = loader.load()
                all_docs.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading document {doc_path}: {e}. Skipping.")
                continue
        
        if not all_docs:
            print("No documents successfully loaded. Aborting ingestion.")
            return

        texts = self.text_splitter.split_documents(all_docs)
        print(f"Split {len(all_docs)} documents into {len(texts)} text chunks.")

        # Add documents to Chroma in batches to avoid potential issues with large lists
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.vector_store.add_documents(batch)
            print(f"Ingested batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

        self.vector_store.persist() # Persist changes to disk
        print("Document ingestion complete and vector store persisted.")
        
        # Re-initialize retriever and chain in case vector store changed significantly (e.g. first ingest)
        self.retriever = self.vector_store.as_retriever(
            search_type=self.vector_store._search_type if hasattr(self.vector_store, '_search_type') else "similarity", # Chroma specific
            search_kwargs=self.vector_store._search_kwargs if hasattr(self.vector_store, '_search_kwargs') else {"k": 3}
        )
        self.conversational_rag_chain = self._build_conversational_rag_chain()
        print("Retriever and conversational RAG chain have been updated.")


    def query(self, question: str, session_id: str = "default_session") -> Optional[str]:
        """Queries the RAG system using the conversational chain with history."""
        if not self.conversational_rag_chain:
            print("Error: Conversational RAG chain not initialized.")
            return "Error: RAG system not fully initialized."

        # Prepare config for RunnableWithMessageHistory
        config: RunnableConfig = {"configurable": {"session_id": session_id}}
        
        try:
            # If MongoDB is not available, the chain is `base_rag_processing_chain`
            # which doesn't inherently use session_id for history.
            # It expects 'question' and optionally 'chat_history' in its input dictionary.
            if not (MONGODB_AVAILABLE and self.memory_config.get("mongodb_connection_string")):
                # For the non-history chain, we need to simulate what RunnableWithMessageHistory does:
                # 1. Load history (empty if no history mechanism)
                # 2. Pass question and history to the chain
                # This part is tricky because the current fallback doesn't manage history.
                # For simplicity, the fallback will answer without history.
                # A more complete solution would require manual history management here if not using MongoDB.
                print("Querying with non-persistent history (fallback).")
                response = self.conversational_rag_chain.invoke(
                    {"question": question, "chat_history": []}, 
                    config=config # config might not be used by base chain, but pass for consistency
                )
            else:
                 # RunnableWithMessageHistory handles history loading and saving automatically
                response = self.conversational_rag_chain.invoke(
                    {"question": question}, # RunnableWithMessageHistory expects the input_messages_key
                    config=config
                )
            return response
        except Exception as e:
            print(f"Error during query invocation: {e}")
            # You might want to log the full traceback here
            return f"An error occurred while processing your query: {str(e)}"


if __name__ == "__main__":
    print("Starting RAG System Example...")

    # --- Configuration ---
    # Important: Set these environment variables before running:
    # export GOOGLE_API_KEY="your_google_api_key"
    # For Azure embeddings (optional, if embedding_provider="azure_openai"):
    # export AZURE_OPENAI_API_KEY="your_azure_openai_key"
    # export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
    # export AZURE_OPENAI_API_VERSION="your_api_version" (e.g., "2023-05-15")
    # export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="your_embedding_deployment_name"
    # For MongoDB chat history (optional):
    # export MONGODB_CONNECTION_STRING="your_mongodb_uri"

    # Choose embedding provider: "huggingface" or "azure_openai"
    EMBEDDING_PROVIDER = "huggingface" # or "azure_openai"
    
    embedding_conf = {}
    if EMBEDDING_PROVIDER == "azure_openai":
        if not all(os.getenv(var) for var in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]):
            print("Azure OpenAI environment variables not set. Exiting.")
            exit(1)
        embedding_conf = {
            "azure_deployment": os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
            # api_key, azure_endpoint, api_version will be picked from env vars by default if not specified here
        }

    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY environment variable not set. Exiting.")
        exit(1)
    
    llm_conf = {
        "google_api_key": os.environ["GOOGLE_API_KEY"],
        "model_name": "gemini-pro", # Or other compatible Gemini model
        "temperature": 0.7
    }

    vector_store_conf = {
        "persist_directory": "./example_chroma_db",
        "collection_name": "example_collection"
    }
    
    # Optional: MongoDB configuration for chat history
    mongodb_conn_str = os.getenv("MONGODB_CONNECTION_STRING")
    memory_conf = {}
    if mongodb_conn_str and MONGODB_AVAILABLE:
        memory_conf = {
            "mongodb_connection_string": mongodb_conn_str,
            "mongodb_database_name": "rag_chat_history_db",
            "mongodb_collection_name": "example_sessions"
        }
        print(f"MongoDB configured for chat history at: {mongodb_conn_str[:30]}...") # Print a truncated URI
    else:
        print("MongoDB connection string not found or pymongo not installed. Chat history will not be persistent.")

    # --- Initialize RAGSystem ---
    try:
        rag_system = RAGSystem(
            embedding_provider=EMBEDDING_PROVIDER,
            embedding_config=embedding_conf,
            llm_config=llm_conf,
            vector_store_config=vector_store_conf,
            memory_config=memory_conf
        )
        print("RAGSystem initialized successfully.")
    except Exception as e:
        print(f"Error initializing RAGSystem: {e}")
        # import traceback
        # traceback.print_exc()
        exit(1)

    # --- Create Sample Documents and Ingest ---
    sample_doc_dir = "rag_example_docs"
    if os.path.exists(sample_doc_dir): # Clean up previous runs for idempotency
        shutil.rmtree(sample_doc_dir)
    os.makedirs(sample_doc_dir, exist_ok=True)

    doc_content = {
        "doc1.txt": "LangChain is a framework for developing applications powered by language models. It simplifies the creation of chains and agents.",
        "doc2.txt": "Gemini is a family of multimodal models by Google. It can understand and generate text, code, images, and more.",
        "doc3.txt": "Retrieval Augmented Generation (RAG) enhances LLM responses by grounding them in external knowledge retrieved from a knowledge base."
    }
    doc_paths = []
    for filename, content in doc_content.items():
        filepath = os.path.join(sample_doc_dir, filename)
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content)
        doc_paths.append(filepath)

    print(f"\nIngesting documents from: {sample_doc_dir}")
    rag_system.ingest_documents(doc_paths)

    # --- Perform Conversational Queries ---
    print("\n--- Starting Conversational Queries ---")
    
    session_1 = "user_session_alpha"
    
    query1 = "What is LangChain?"
    print(f"\nSession: {session_1}, Query: {query1}")
    response1 = rag_system.query(query1, session_id=session_1)
    print(f"Response: {response1}")

    query2 = "And what about RAG?" # Follow-up question
    print(f"\nSession: {session_1}, Query (follow-up): {query2}")
    response2 = rag_system.query(query2, session_id=session_1)
    print(f"Response: {response2}")
    
    query3 = "Tell me more about Gemini." # New topic in the same session
    print(f"\nSession: {session_1}, Query: {query3}")
    response3 = rag_system.query(query3, session_id=session_1)
    print(f"Response: {response3}")

    # --- Example of a different session (to show history is isolated) ---
    if MONGODB_AVAILABLE and memory_conf.get("mongodb_connection_string"): # Only if history is persistent
        session_2 = "user_session_beta"
        query4 = "What is LangChain?" # Same question as query1, but different session
        print(f"\nSession: {session_2}, Query: {query4}")
        response4 = rag_system.query(query4, session_id=session_2)
        print(f"Response: {response4} \n(Note: This response should not use context from session_1's RAG question, but rephrasing might occur if there was prior chat in session_2)")
    
    # --- Clean up ---
    print(f"\nCleaning up sample document directory: {sample_doc_dir}")
    shutil.rmtree(sample_doc_dir)
    # Note: ChromaDB data in vector_store_conf["persist_directory"] is not cleaned up automatically by this script.
    print(f"ChromaDB data remains at: {vector_store_conf['persist_directory']}")
    print("\nExample script finished.")
```
