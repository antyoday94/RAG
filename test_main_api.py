import unittest
import os
import datetime
import json # Added for parsing SSE data
import asyncio # Added for async helper
from unittest.mock import patch, MagicMock, ANY

from fastapi.testclient import TestClient

# Import the FastAPI app instance from main.py
try:
    from main import app # Assuming Pydantic models are also in main or not needed for these tests' assertions
    MAIN_APP_AVAILABLE = True
except ImportError as e:
    print(f"Error importing main app or components: {e}. API tests will be skipped.")
    MAIN_APP_AVAILABLE = False
    app = None # Placeholder


# Mock environment variables for the entire test class
@patch.dict(os.environ, {
    "GOOGLE_API_KEY": "test_mock_google_api_key",
    "EMBEDDING_PROVIDER": "huggingface", 
    "HF_EMBEDDING_MODEL_NAME": "mock-hf-model",
    "VECTOR_STORE_PERSIST_DIRECTORY": "./test_api_chroma_db_stream", # Updated path
    "MONGODB_CONNECTION_STRING": "", 
    "LLM_TEMPERATURE": "0.0" 
}, clear=True)
@unittest.skipIf(not MAIN_APP_AVAILABLE, "Skipping API tests because main.app could not be imported.")
class TestMainAPIStreaming(unittest.TestCase): # Renamed class for clarity

    mock_rag_system_class_patcher = None
    MockRAGSystemClass = None 

    @classmethod
    def setUpClass(cls):
        cls.mock_rag_system_class_patcher = patch('main.RAGSystem', KEEPS_AUTO_SPEC=True)
        cls.MockRAGSystemClass = cls.mock_rag_system_class_patcher.start()
        
        db_path = "./test_api_chroma_db_stream" # Match env var
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)

    @classmethod
    def tearDownClass(cls):
        if cls.mock_rag_system_class_patcher:
            cls.mock_rag_system_class_patcher.stop()
        
        db_path = "./test_api_chroma_db_stream" # Match env var
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)

    def setUp(self):
        self.MockRAGSystemClass.reset_mock() 
        self.mock_rag_instance = self.MockRAGSystemClass.return_value 
        # Ensure astream_query is a MagicMock for async iteration behavior
        self.mock_rag_instance.astream_query = MagicMock()

        self.client = TestClient(app)
        
        import main 
        main.rag_system_instance = self.mock_rag_instance

    def tearDown(self):
        import main
        main.rag_system_instance = None

    async def mock_astream_query_generator(self, chunks: list, error_after_chunks: Exception = None):
        """Async generator to mock RAGSystem's astream_query."""
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0) # Ensure it behaves like a real async generator
        if error_after_chunks:
            raise error_after_chunks

    # --- /health/ Endpoint Tests (largely unchanged but verified) ---
    def test_health_check_rag_initialized_and_healthy(self):
        with patch('main.RAG_CORE_IMPORTED', True), \
             patch('main.GOOGLE_GENAI_AVAILABLE', True), \
             patch('main.AZURE_OPENAI_AVAILABLE', True), \
             patch('main.MONGODB_AVAILABLE', True):
            response = self.client.get("/health/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["rag_system_initialized"])
        self.assertTrue(data["rag_core_imported"])

    def test_health_check_rag_initialization_failed(self):
        import main
        main.rag_system_instance = None 
        with patch('main.RAG_CORE_IMPORTED', True):
            response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "unavailable")
        self.assertFalse(data["rag_system_initialized"])

    def test_health_check_rag_core_not_imported(self):
        import main
        main.rag_system_instance = None
        with patch('main.RAG_CORE_IMPORTED', False): 
            response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "unavailable")
        self.assertFalse(data["rag_core_imported"])

    # --- /query/ Endpoint Tests (Updated for Streaming) ---
    def test_query_streaming_success(self):
        """Test a successful streaming query to /query/."""
        expected_chunks = ["Hello", ", ", "world", "!"]
        expected_full_message = "".join(expected_chunks)
        
        # Configure the mock astream_query to use our async generator
        self.mock_rag_instance.astream_query.return_value = self.mock_astream_query_generator(expected_chunks)
        
        payload = {"query_text": "Say hello", "session_id": "test_stream_success"}
        
        response = self.client.post("/query/", json=payload) # TestClient handles streaming responses
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream")

        received_tokens = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                try:
                    data_json = json.loads(line.split("data: ", 1)[1])
                    self.assertIn("token", data_json)
                    received_tokens.append(data_json["token"])
                except json.JSONDecodeError:
                    self.fail(f"Failed to parse JSON from SSE data line: {line}")
        
        reconstructed_message = "".join(received_tokens)
        self.assertEqual(reconstructed_message, expected_full_message)
        
        self.mock_rag_instance.astream_query.assert_called_once_with(
            question="Say hello",
            session_id="test_stream_success"
        )

    def test_query_streaming_error_midstream(self):
        """Test /query/ when RAG system's astream_query raises an error mid-stream."""
        initial_chunks = ["First ", "part. "]
        error_message = "RAG stream failed!"
        
        self.mock_rag_instance.astream_query.return_value = self.mock_astream_query_generator(
            initial_chunks, error_after_chunks=Exception(error_message)
        )
        
        payload = {"query_text": "This will error mid-stream", "session_id": "test_stream_error"}
        response = self.client.post("/query/", json=payload)
        
        self.assertEqual(response.status_code, 200) # Connection is established, error is in stream
        self.assertEqual(response.headers["content-type"], "text/event-stream")

        received_tokens = []
        error_event_received = False
        error_detail = ""

        lines = list(response.iter_lines()) # Get all lines to inspect them
        
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("data: "):
                data_json = json.loads(line.split("data: ", 1)[1])
                received_tokens.append(data_json["token"])
            elif line.startswith("event: error"):
                error_event_received = True
                if idx + 1 < len(lines) and lines[idx+1].startswith("data: "):
                    error_data_json = json.loads(lines[idx+1].split("data: ", 1)[1])
                    # As per main.py, error detail is in 'token' field of StreamToken for errors too
                    error_detail = error_data_json.get("token") 
                    idx += 1 # Consume the data line for the error event
            idx += 1
            
        self.assertEqual("".join(received_tokens), "".join(initial_chunks))
        self.assertTrue(error_event_received, "SSE error event was not received.")
        # Check if the error message from the exception is in the detail
        # The main.py formats it as: f"An error occurred while processing your streaming query: {str(e)}"
        self.assertIn(error_message, error_detail) 
        self.assertIn("An error occurred while processing your streaming query", error_detail)

        self.mock_rag_instance.astream_query.assert_called_once_with(
            question="This will error mid-stream",
            session_id="test_stream_error"
        )

    # --- Non-Streaming Error Tests (should still pass) ---
    def test_query_missing_query_text(self):
        payload = {"session_id": "test_session_missing_text"}
        response = self.client.post("/query/", json=payload)
        self.assertEqual(response.status_code, 422) 

    def test_query_rag_system_not_initialized(self):
        import main
        main.rag_system_instance = None 
        
        payload = {"query_text": "Hello?", "session_id": "test_session_503"}
        response = self.client.post("/query/", json=payload)
        self.assertEqual(response.status_code, 503)
        self.assertIn("RAG System is not initialized", response.json()["detail"])

    # The following test needs to be adapted if /query/ only streams.
    # If the previous non-streaming error (e.g. RAGSystem.query raises Exception)
    # is now handled by the stream_generator, then the response will be a stream with an error event.
    # However, the task is to test errors *before* streaming begins.
    # The `test_query_rag_system_not_initialized` covers one such case.
    # Let's assume `main.rag_system_instance.astream_query` itself could raise an immediate error
    # before yielding anything, which would be caught by the FastAPI endpoint wrapper.
    # This test is less about mid-stream errors and more about initial call failure.
    def test_query_astream_query_immediate_exception(self):
        """Test /query/ when rag_system_instance.astream_query itself raises an exception immediately."""
        # This simulates an error in astream_query before it even starts yielding (e.g., bad input to the method, not a mid-stream LLM error)
        # The stream_generator in main.py has a try-except that would catch this and yield an SSE error.
        error_message = "Immediate astream_query error"
        self.mock_rag_instance.astream_query.side_effect = Exception(error_message)
        
        payload = {"query_text": "This will fail immediately.", "session_id": "test_session_astream_immediate_fail"}
        response = self.client.post("/query/", json=payload)
        
        self.assertEqual(response.status_code, 200) # Connection established
        self.assertEqual(response.headers["content-type"], "text/event-stream")

        error_event_received = False
        error_detail = ""
        for line in response.iter_lines():
            if line.startswith("event: error"):
                error_event_received = True
            elif line.startswith("data: ") and error_event_received: # data line for the error event
                error_data_json = json.loads(line.split("data: ", 1)[1])
                error_detail = error_data_json.get("token")
                break # Found the error event and its data
        
        self.assertTrue(error_event_received)
        self.assertIn(error_message, error_detail)
        self.assertIn("An error occurred while processing your streaming query", error_detail)


if __name__ == '__main__':
    if MAIN_APP_AVAILABLE:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        print("Skipping API tests execution as main.app could not be imported.")

```
