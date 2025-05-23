import os
import shutil
import time # Added for unique session IDs
from typing import List, Optional

# Ensure rag_core can be imported.
# This assumes rag_core.py is in the same directory or PYTHONPATH
try:
    from rag_core import RAGSystem
    RAG_CORE_IMPORTED = True
except ImportError as e:
    RAG_CORE_IMPORTED = False
    RAGSystem = None # Placeholder
    print(f"Failed to import RAGSystem from rag_core: {e}. Script functionality will be limited.")

# --- Constants for Sample Data (V2) ---
DATA_DIR_V2 = "industrial_sample_data_v2"
TECHNICAL_MANUAL_FILE_V2 = os.path.join(DATA_DIR_V2, "tech_manual_v2.txt")
HR_POLICY_FILE_V2 = os.path.join(DATA_DIR_V2, "hr_policy_v2.txt")
CODE_EXPLANATION_FILE_V2 = os.path.join(DATA_DIR_V2, "simple_code_explanation_v2_py_as_txt.txt")

ALL_SAMPLE_FILES_V2 = [TECHNICAL_MANUAL_FILE_V2, HR_POLICY_FILE_V2, CODE_EXPLANATION_FILE_V2]

# Content for the sample files (remains the same)
TECHNICAL_MANUAL_CONTENT = """
Section 1: Troubleshooting Error Code E001
If you encounter error code E001, it typically indicates a sensor malfunction.
Steps to resolve:
1. Power cycle the device.
2. Check sensor cable connections for any damage or loose fittings.
3. If the error persists, replace sensor module (Part #SNSR-001).
Contact support if replacement does not resolve the issue.

Section 2: Maintenance Schedule
Perform routine maintenance every 3 months.
Maintenance includes:
- Cleaning air filters.
- Lubricating moving parts as specified in Appendix A.
- Calibrating sensor array.
"""

HR_POLICY_CONTENT = """
Company HR Policy Document

Section 1: Remote Work
Employees may be eligible for remote work based on their role and manager approval.
A formal request must be submitted through the HR portal.
Remote employees must maintain a dedicated and safe workspace.

Section 2: Leave Policy
Annual leave: 20 days per year for full-time employees.
Sick leave: 10 days per year, with a doctor's note required for absences longer than 2 days.
Parental leave: Up to 12 weeks of unpaid leave.
"""

CODE_EXPLANATION_CONTENT = """
def greet(name):
  \"\"\"This function greets the person passed in as a parameter.\"\"\"
  return f"Hello, {name}!"

def calculate_area(length, width):
  \"\"\"This function calculates the area of a rectangle.\"\"\"
  if length < 0 or width < 0:
    raise ValueError("Length and width must be non-negative.")
  return length * width
"""

SAMPLE_DATA_V2 = {
    TECHNICAL_MANUAL_FILE_V2: TECHNICAL_MANUAL_CONTENT,
    HR_POLICY_FILE_V2: HR_POLICY_CONTENT,
    CODE_EXPLANATION_FILE_V2: CODE_EXPLANATION_CONTENT,
}

# --- Helper Functions ---
def setup_sample_data_v2():
    """Creates the V2 sample data directory and files."""
    if os.path.exists(DATA_DIR_V2): # Clean up previous V2 runs for idempotency
        shutil.rmtree(DATA_DIR_V2)
    os.makedirs(DATA_DIR_V2, exist_ok=True)
    print(f"Created directory: {DATA_DIR_V2}")

    for filepath, content in SAMPLE_DATA_V2.items():
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content)
        # print(f"Created sample file: {filepath}")
    print(f"Populated {len(SAMPLE_DATA_V2)} sample files in {DATA_DIR_V2}.")


def cleanup_sample_data_v2():
    """Removes the V2 sample data directory and files."""
    if os.path.exists(DATA_DIR_V2):
        shutil.rmtree(DATA_DIR_V2)
        print(f"Removed directory and its contents: {DATA_DIR_V2}")

def check_env_vars(required_vars: List[str]) -> bool:
    """Checks if all required environment variables are set."""
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("\n--- Environment Variable Check ---")
        for var in missing_vars:
            print(f"Warning: Environment variable '{var}' is not set.")
        print("Please set these variables to run all configured examples.")
        print("--- End Environment Variable Check ---\n")
        return False
    return True

def run_demonstrations_v2(rag_system_instance: RAGSystem, example_type: str):
    """Runs a series of conversational demonstration queries."""
    print(f"\n--- Running Demonstrations for: {example_type} ---")

    # Technical Support Conversation
    tech_session_id = f"tech_support_{time.time_ns()}"
    print(f"\n-- Technical Support (Session ID: {tech_session_id}) --")
    
    q1_tech = "What should I do if I see error code E001?"
    print(f"Query: {q1_tech}")
    response = rag_system_instance.query(q1_tech, session_id=tech_session_id)
    print(f"Response: {response}\n")

    q2_tech = "And what is the part number for the sensor module mentioned?"
    print(f"Query (follow-up): {q2_tech}")
    response = rag_system_instance.query(q2_tech, session_id=tech_session_id)
    print(f"Response: {response}\n")

    q3_tech = "How often is routine maintenance recommended?" # New question in same session
    print(f"Query: {q3_tech}")
    response = rag_system_instance.query(q3_tech, session_id=tech_session_id)
    print(f"Response: {response}\n")

    # HR Policy Conversation
    hr_session_id = f"hr_policy_{time.time_ns()}"
    print(f"\n-- HR Policy Q&A (Session ID: {hr_session_id}) --")

    q1_hr = "How many days of annual leave do full-time employees get?"
    print(f"Query: {q1_hr}")
    response = rag_system_instance.query(q1_hr, session_id=hr_session_id)
    print(f"Response: {response}\n")

    q2_hr = "What about remote work?"
    print(f"Query (follow-up): {q2_hr}")
    response = rag_system_instance.query(q2_hr, session_id=hr_session_id)
    print(f"Response: {response}\n")

    # Code Explanation Conversation
    code_session_id = f"code_explain_{time.time_ns()}"
    print(f"\n-- Code Explanation (Session ID: {code_session_id}) --")
    
    q1_code = "What does the 'greet' function do?"
    print(f"Query: {q1_code}")
    response = rag_system_instance.query(q1_code, session_id=code_session_id)
    print(f"Response: {response}\n")

    q2_code = "And how does 'calculate_area' handle negative inputs?"
    print(f"Query (follow-up): {q2_code}")
    response = rag_system_instance.query(q2_code, session_id=code_session_id)
    print(f"Response: {response}\n")

    print(f"--- Finished Demonstrations for: {example_type} ---")


if __name__ == "__main__":
    if not RAG_CORE_IMPORTED:
        print("Exiting script as RAGSystem could not be imported from rag_core.py.")
        exit(1)

    print("===================================================")
    print(" Starting RAG Example Script (V2 with Conversations) ")
    print("===================================================")
    print("IMPORTANT: This script uses Google Generative AI for LLM operations.")
    print("Ensure 'GOOGLE_API_KEY' environment variable is set.")
    print("Optional: Set 'MONGODB_CONNECTION_STRING' for persistent chat history.")
    print("Optional: Set Azure OpenAI env vars for the Azure embeddings example.\n")

    # --- Common Setup ---
    setup_sample_data_v2()
    
    # --- General LLM and Memory Config (can be shared) ---
    google_api_key_set = check_env_vars(["GOOGLE_API_KEY"])
    if not google_api_key_set:
        print("GOOGLE_API_KEY is not set. LLM functionalities will fail. Aborting.")
        cleanup_sample_data_v2()
        exit(1)

    llm_config_google = {
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "model_name": "gemini-pro",
        "temperature": 0.1 # Lower temperature for more factual Q&A from RAG
    }
    
    mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    memory_config_mongo = {}
    if mongodb_connection_string:
        print(f"MongoDB connection string found. Chat history will be persistent.")
        memory_config_mongo = {
            "mongodb_connection_string": mongodb_connection_string,
            "mongodb_database_name": "rag_chat_history_examples_v2", # Use a specific DB name
            "mongodb_collection_name": "example_sessions_v2"
        }
    else:
        print("MONGODB_CONNECTION_STRING not set. Chat history will be in-memory for each session (if chain supports) or non-existent.")

    # --- HuggingFace Embeddings Example ---
    print("\n===================================================")
    print(" Example 1: HuggingFace Embeddings & Google LLM ")
    print("===================================================")
    rag_system_hf = None
    try:
        embedding_config_hf = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {'device': 'cpu'}
        }
        vector_store_config_hf = {
            "persist_directory": "./chroma_db_examples_hf", # Separate DB for HF
            "collection_name": "industrial_hf_v2"
        }
        
        print("Initializing RAGSystem with HuggingFace Embeddings...")
        rag_system_hf = RAGSystem(
            embedding_provider="huggingface",
            embedding_config=embedding_config_hf,
            llm_provider="google",
            llm_config=llm_config_google,
            vector_store_config=vector_store_config_hf,
            memory_config=memory_config_mongo
        )
        print("RAGSystem (HF) initialized.")

        print(f"Ingesting documents for HF example: {ALL_SAMPLE_FILES_V2}")
        rag_system_hf.ingest_documents(ALL_SAMPLE_FILES_V2)
        print("Document ingestion for HF example complete.")

        run_demonstrations_v2(rag_system_hf, "HuggingFace Embeddings & Google LLM")

    except Exception as e:
        print(f"An error occurred during the HuggingFace Embeddings example: {e}")
        # import traceback
        # traceback.print_exc()
    finally:
        if rag_system_hf:
             print(f"Note: ChromaDB data for HuggingFace example remains at: {rag_system_hf.vector_store.persist_directory}")


    # --- Azure OpenAI Embeddings Example ---
    print("\n===================================================")
    print(" Example 2: Azure OpenAI Embeddings & Google LLM ")
    print("===================================================")
    
    azure_env_vars = [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION"
    ]
    azure_vars_set = check_env_vars(azure_env_vars)
    rag_system_azure = None

    if azure_vars_set:
        try:
            embedding_config_azure = {
                "azure_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            }
            vector_store_config_azure = {
                "persist_directory": "./chroma_db_examples_azure", # Separate DB for Azure
                "collection_name": "industrial_azure_v2"
            }

            print("Initializing RAGSystem with Azure OpenAI Embeddings...")
            rag_system_azure = RAGSystem(
                embedding_provider="azure_openai",
                embedding_config=embedding_config_azure,
                llm_provider="google", # Still using Google for LLM part
                llm_config=llm_config_google,
                vector_store_config=vector_store_config_azure,
                memory_config=memory_config_mongo # Can reuse memory config
            )
            print("RAGSystem (Azure) initialized.")

            # Re-ingest documents into the new Azure-backed vector store
            # This is important because the embeddings are different.
            print(f"Ingesting documents for Azure example: {ALL_SAMPLE_FILES_V2}")
            rag_system_azure.ingest_documents(ALL_SAMPLE_FILES_V2)
            print("Document ingestion for Azure example complete.")

            run_demonstrations_v2(rag_system_azure, "Azure OpenAI Embeddings & Google LLM")

        except Exception as e:
            print(f"An error occurred during the Azure OpenAI Embeddings example: {e}")
            # import traceback
            # traceback.print_exc()
        finally:
            if rag_system_azure:
                print(f"Note: ChromaDB data for Azure example remains at: {rag_system_azure.vector_store.persist_directory}")
    else:
        print("Azure OpenAI environment variables not fully set. Skipping Azure OpenAI Embeddings example.")

    # --- Common Cleanup ---
    finally:
        cleanup_sample_data_v2()
        print("\n===================================================")
        print(" RAG Example Script (V2) Finished ")
        print("===================================================")
```
