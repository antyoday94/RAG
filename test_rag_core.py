import unittest
import os
import shutil
from unittest.mock import patch, MagicMock, Mock, ANY, call

# Attempt to import RAGSystem and other necessary components from rag_core
# This allows tests to be discovered even if rag_core or its dependencies are not fully available
try:
    from rag_core import RAGSystem, format_docs
    from rag_core import GOOGLE_GENAI_AVAILABLE, AZURE_OPENAI_AVAILABLE, MONGODB_AVAILABLE
    RAG_CORE_FULLY_AVAILABLE = GOOGLE_GENAI_AVAILABLE and AZURE_OPENAI_AVAILABLE and MONGODB_AVAILABLE
except ImportError:
    # Define dummy/placeholder classes if rag_core or its components can't be imported
    class RAGSystem: pass
    def format_docs(docs): return ""
    GOOGLE_GENAI_AVAILABLE = False
    AZURE_OPENAI_AVAILABLE = False
    MONGODB_AVAILABLE = False
    RAG_CORE_FULLY_AVAILABLE = False

# Import Langchain message types, define dummies if langchain_core is not available
try:
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    class Document:
        def __init__(self, page_content, metadata=None): self.page_content = page_content; self.metadata = metadata or {}
    class AIMessage:
        def __init__(self, content): self.content = content
    class HumanMessage:
        def __init__(self, content): self.content = content
    class SystemMessage: # Not directly used in these tests' mocks but good for completeness
        def __init__(self, content): self.content = content
    LANGCHAIN_CORE_AVAILABLE = False

# --- Test Configuration ---
TEST_DOCS_DIR_V2 = "test_sample_docs_v2"
TEST_TXT_FILE_V2 = os.path.join(TEST_DOCS_DIR_V2, "test_doc_v2.txt")

TEST_CHROMA_DB_DIR_HF = "./test_chroma_db_hf"
TEST_CHROMA_DB_DIR_AZURE = "./test_chroma_db_azure"
TEST_CHROMA_DB_DIR_NOMEM = "./test_chroma_db_nomem"


# Decorator to skip tests if core dependencies are missing
# Adjust this condition based on which parts of rag_core are truly essential for the tests to run
# For example, if only GOOGLE_GENAI_AVAILABLE is critical for most tests because Azure/Mongo are optional features.
skip_if_dependencies_unavailable = unittest.skipIf(
    not (RAG_CORE_FULLY_AVAILABLE and LANGCHAIN_CORE_AVAILABLE),
    "Skipping tests due to missing rag_core or langchain_core dependencies."
)

# --- Main Test Class ---
@skip_if_dependencies_unavailable
@patch.dict(os.environ, {
    "GOOGLE_API_KEY": "test_google_api_key",
    "AZURE_OPENAI_API_KEY": "test_azure_api_key",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "test-embedding-deployment",
    "AZURE_OPENAI_API_VERSION": "2023-05-15",
    "MONGODB_CONNECTION_STRING": "mongodb://test:27017/testdb"
})
@patch('rag_core.HuggingFaceEmbeddings')
@patch('rag_core.AzureOpenAIEmbeddings')
@patch('rag_core.Chroma')
@patch('rag_core.ChatGoogleGenerativeAI')
@patch('rag_core.MongoDBChatMessageHistory')
@patch('rag_core.TextLoader') # For ingest_documents
class TestRAGSystemV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create dummy document directory and file
        os.makedirs(TEST_DOCS_DIR_V2, exist_ok=True)
        with open(TEST_TXT_FILE_V2, "w") as f:
            f.write("This is a V2 test document about LangChain and AI.")
        
        # Ensure test ChromaDB directories are clean before tests
        for db_dir in [TEST_CHROMA_DB_DIR_HF, TEST_CHROMA_DB_DIR_AZURE, TEST_CHROMA_DB_DIR_NOMEM]:
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)

    @classmethod
    def tearDownClass(cls):
        # Clean up dummy document directory
        if os.path.exists(TEST_DOCS_DIR_V2):
            shutil.rmtree(TEST_DOCS_DIR_V2)
        # Clean up ChromaDB test directories
        for db_dir in [TEST_CHROMA_DB_DIR_HF, TEST_CHROMA_DB_DIR_AZURE, TEST_CHROMA_DB_DIR_NOMEM]:
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)

    def setUp(self, MockTextLoader, MockMongoDBChatMessageHistory, MockChatGoogleGenerativeAI,
              MockChroma, MockAzureOpenAIEmbeddings, MockHuggingFaceEmbeddings):
        """Set up mocks for each test."""
        self.mock_hf_embeddings_class = MockHuggingFaceEmbeddings
        self.mock_azure_embeddings_class = MockAzureOpenAIEmbeddings
        self.mock_chroma_class = MockChroma
        self.mock_chat_google_genai_class = MockChatGoogleGenerativeAI
        self.mock_mongodb_chat_history_class = MockMongoDBChatMessageHistory
        self.mock_text_loader_class = MockTextLoader

        # Configure default mock instances
        self.mock_hf_embeddings_instance = self.mock_hf_embeddings_class.return_value
        self.mock_azure_embeddings_instance = self.mock_azure_embeddings_class.return_value
        
        self.mock_chroma_instance = self.mock_chroma_class.return_value
        self.mock_retriever_instance = self.mock_chroma_instance.as_retriever.return_value
        self.mock_retriever_instance.invoke.return_value = [Document(page_content="mocked retrieved context")]
        
        # Setup ChatGoogleGenerativeAI to return distinct mocks for rag and rephrase
        self.mock_gemini_instance_rag = MagicMock(spec=MockChatGoogleGenerativeAI.return_value)
        self.mock_gemini_instance_rag.invoke.return_value = AIMessage(content="Mocked RAG response")
        
        self.mock_gemini_instance_rephrase = MagicMock(spec=MockChatGoogleGenerativeAI.return_value)
        self.mock_gemini_instance_rephrase.invoke.return_value = AIMessage(content="Mocked rephrased question")

        # Use side_effect to control which mock is returned by ChatGoogleGenerativeAI constructor
        # The first call to ChatGoogleGenerativeAI in RAGSystem is for llm_rag, second for llm_rephrase
        self.mock_chat_google_genai_class.side_effect = [
            self.mock_gemini_instance_rag, 
            self.mock_gemini_instance_rephrase
        ]

        self.mock_mongodb_chat_history_instance = self.mock_mongodb_chat_history_class.return_value
        self.mock_mongodb_chat_history_instance.messages = [] # Default to no history

        self.mock_text_loader_instance = self.mock_text_loader_class.return_value
        self.mock_text_loader_instance.load.return_value = [Document(page_content="Loaded test document content.")]

        # Reset call counts for class-level mocks if they are being reused across tests in a way that accumulates calls
        # For mocks created per test (like instance mocks above), this is not usually necessary.
        self.mock_hf_embeddings_class.reset_mock()
        self.mock_azure_embeddings_class.reset_mock()
        self.mock_chroma_class.reset_mock()
        self.mock_chat_google_genai_class.reset_mock()
        self.mock_mongodb_chat_history_class.reset_mock()
        self.mock_text_loader_class.reset_mock()


    def test_initialization_huggingface_mongodb(self, _ML, _MCH, _MCGAI, _MC, _MAOE, _MHFE):
        """Test RAGSystem initialization with HuggingFace, Google LLM, and MongoDB."""
        embedding_config_hf = {"model_name": "test-hf-model"}
        llm_config_google = {"model_name": "gemini-test", "temperature": 0.1}
        vector_store_config_hf = {"persist_directory": TEST_CHROMA_DB_DIR_HF}
        memory_config_mongo = {
            "mongodb_connection_string": os.environ["MONGODB_CONNECTION_STRING"],
            "mongodb_database_name": "test_db",
            "mongodb_collection_name": "test_coll"
        }

        RAGSystem(
            embedding_provider="huggingface", embedding_config=embedding_config_hf,
            llm_provider="google", llm_config=llm_config_google,
            vector_store_config=vector_store_config_hf, memory_config=memory_config_mongo
        )

        _MHFE.assert_called_once_with(model_name="test-hf-model", model_kwargs=ANY)
        _MCGAI.assert_any_call(model="gemini-test", google_api_key="test_google_api_key", temperature=0.1, convert_system_message_to_human=True) # RAG
        _MCGAI.assert_any_call(model="gemini-test", google_api_key="test_google_api_key", temperature=0.0, convert_system_message_to_human=True) # Rephrase
        _MC.assert_called_once_with(
            embedding_function=_MHFE.return_value,
            persist_directory=TEST_CHROMA_DB_DIR_HF,
            collection_name=ANY
        )
        # MongoDBChatMessageHistory is instantiated dynamically within the chain, so we check later during query.


    def test_initialization_azure_openai_no_mongodb(self, _ML, _MCH, _MCGAI, _MC, _MAOE, _MHFE):
        """Test RAGSystem initialization with Azure OpenAI and no MongoDB."""
        embedding_config_azure = {"azure_deployment": "test-azure-deployment"}
        llm_config_google = {"model_name": "gemini-pro"}
        vector_store_config_azure = {"persist_directory": TEST_CHROMA_DB_DIR_AZURE}
        
        RAGSystem(
            embedding_provider="azure_openai", embedding_config=embedding_config_azure,
            llm_provider="google", llm_config=llm_config_google,
            vector_store_config=vector_store_config_azure, memory_config={} # No MongoDB
        )

        _MAOE.assert_called_once_with(
            azure_deployment="test-azure-deployment", api_key="test_azure_api_key",
            azure_endpoint="https://test.openai.azure.com/", api_version="2023-05-15"
        )
        _MCGAI.call_count == 2 # For RAG and rephrase
        _MC.assert_called_once_with(
            embedding_function=_MAOE.return_value,
            persist_directory=TEST_CHROMA_DB_DIR_AZURE,
            collection_name=ANY
        )
        _MCH.assert_not_called() # MongoDBChatMessageHistory should not be called if not configured

    def test_ingest_documents_chroma(self, _ML, _MCH, _MCGAI, _MC, _MAOE, _MHFE):
        """Test document ingestion with ChromaDB."""
        rag_system = RAGSystem(vector_store_config={"persist_directory": TEST_CHROMA_DB_DIR_HF})
        
        rag_system.ingest_documents([TEST_TXT_FILE_V2])

        self.mock_text_loader_class.assert_called_once_with(TEST_TXT_FILE_V2, encoding='utf-8')
        self.mock_text_loader_instance.load.assert_called_once()
        
        # Ensure documents are added to Chroma and persisted
        self.mock_chroma_instance.add_documents.assert_called_once_with(ANY) # ANY checks for the list of Document objects
        self.mock_chroma_instance.persist.assert_called_once()


    def test_query_no_history(self, _ML, _MCH, _MCGAI, _MC, _MAOE, _MHFE):
        """Test query when no chat history is present (rephrasing LLM should not be called)."""
        rag_system = RAGSystem(memory_config={}) # No MongoDB, so no history retrieval by default
        
        test_question = "What is LangChain?"
        response = rag_system.query(test_question, session_id="test_session_no_hist")

        self.mock_gemini_instance_rephrase.invoke.assert_not_called()
        self.mock_retriever_instance.invoke.assert_called_once_with(test_question)
        self.mock_gemini_instance_rag.invoke.assert_called_once() # Check specific args if needed
        self.assertEqual(response, "Mocked RAG response")


    def test_query_with_history_mongodb(self, _ML, _MCH, _MCGAI, _MC, _MAOE, _MHFE):
        """Test query with chat history from MongoDB."""
        memory_conf = {
            "mongodb_connection_string": os.environ["MONGODB_CONNECTION_STRING"],
            "mongodb_database_name": "test_rag_history_db",
            "mongodb_collection_name": "test_rag_sessions"
        }
        rag_system = RAGSystem(memory_config=memory_conf)

        # Simulate existing chat history
        self.mock_mongodb_chat_history_instance.messages = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer")
        ]
        
        test_question = "More details please."
        rephrased_question_content = "Standalone: More details please about previous topic."
        self.mock_gemini_instance_rephrase.invoke.return_value = AIMessage(content=rephrased_question_content)

        response = rag_system.query(test_question, session_id="test_session_with_hist")

        _MCH.assert_called_with( # Check constructor call for MongoDBChatMessageHistory
            session_id="test_session_with_hist",
            connection_string=os.environ["MONGODB_CONNECTION_STRING"],
            database_name="test_rag_history_db",
            collection_name="test_rag_sessions"
        )
        self.mock_gemini_instance_rephrase.invoke.assert_called_once() # With specific prompt if needed
        self.mock_retriever_instance.invoke.assert_called_once_with(rephrased_question_content)
        self.mock_gemini_instance_rag.invoke.assert_called_once() # Check it received rephrased_question
        # Example of checking part of the prompt passed to RAG LLM:
        args, kwargs = self.mock_gemini_instance_rag.invoke.call_args
        prompt_arg = args[0] # Assuming the prompt is the first positional argument
                        
        # Check that the rephrased question is in the user message part of the prompt
        found_rephrased_in_prompt = False
        for message in prompt_arg.messages: # prompt_arg is a ChatPromptValue
            if isinstance(message, HumanMessage) and rephrased_question_content in message.content:
                found_rephrased_in_prompt = True
                break
        self.assertTrue(found_rephrased_in_prompt, "Rephrased question not found in RAG prompt's HumanMessage")

        self.assertEqual(response, "Mocked RAG response")


    def test_query_no_mongodb_configured_fallback(self, _ML, _MCH, _MCGAI, _MC, _MAOE, _MHFE):
        """Test query behavior when MongoDB is not configured (fallback to no history)."""
        rag_system = RAGSystem(memory_config={}) # Explicitly no MongoDB config

        test_question = "What about AI safety?"
        response = rag_system.query(test_question, session_id="test_session_no_mongo_fallback")

        # MongoDBChatMessageHistory class itself should not be instantiated by RunnableWithMessageHistory
        # if the factory function returns None or if the branch for MongoDB is not taken.
        # In our rag_core, if mongo is not configured, RunnableWithMessageHistory is not even used for the main chain.
        # So, the constructor for MongoDBChatMessageHistory should not be called AT ALL by the chain.
        _MCH.assert_not_called() 
        
        # Rephrasing LLM should not be called as no chat_history is passed to the base chain in this fallback
        self.mock_gemini_instance_rephrase.invoke.assert_not_called()
        
        self.mock_retriever_instance.invoke.assert_called_once_with(test_question)
        self.mock_gemini_instance_rag.invoke.assert_called_once()
        self.assertEqual(response, "Mocked RAG response")

    def test_format_docs_utility(self, _ML, _MCH, _MCGAI, _MC, _MAOE, _MHFE):
        """Test the format_docs utility function."""
        docs = [
            Document(page_content="First document content."),
            Document(page_content="Second document content.")
        ]
        self.assertEqual(format_docs(docs), "First document content.\n\nSecond document content.")


if __name__ == '__main__':
    if RAG_CORE_FULLY_AVAILABLE and LANGCHAIN_CORE_AVAILABLE:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        print("Skipping RAG Core V2 tests as one or more core dependencies (rag_core, langchain_core, or their sub-components like Google/Azure/Mongo availability flags) are not available.")
        print(f"Status: RAG_CORE_FULLY_AVAILABLE={RAG_CORE_FULLY_AVAILABLE}, LANGCHAIN_CORE_AVAILABLE={LANGCHAIN_CORE_AVAILABLE}")
        print(f"Breakdown: GOOGLE_GENAI_AVAILABLE={GOOGLE_GENAI_AVAILABLE}, AZURE_OPENAI_AVAILABLE={AZURE_OPENAI_AVAILABLE}, MONGODB_AVAILABLE={MONGODB_AVAILABLE}")

```
