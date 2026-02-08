import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Base Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # PDF Source
    # Default to the path found in the user's workspace or override via env var
    BNS_PDF_PATH = os.getenv("BNS_PDF_PATH", DATA_DIR / "raw" / "BNS.pdf")
    
    # Processed Data
    PROCESSED_DIR = DATA_DIR / "processed"
    BNS_TEXT_JSON = PROCESSED_DIR / "bns_text.json"
    BNS_CHUNKS_JSON = PROCESSED_DIR / "bns_chunks.json"
    
    # Vector Store
    VECTOR_STORE_DIR = DATA_DIR / "vector_store"
    COLLECTION_NAME = "bns_sections"
    
    # Mappings
    MAPPINGS_DIR = DATA_DIR / "mappings"
    IPC_BNS_CSV = MAPPINGS_DIR / "ipc_bns_mapping.csv"
    
    # Embeddings: Using sentence-transformers (runs locally - FREE!)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight, free
    
    # LLM: Using Groq API (High Speed!)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    # Legacy Gemini support
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Legacy Ollama support (if you want to switch back)
    # OLLAMA_MODEL = "llama3.2"
    # OLLAMA_BASE_URL = "http://localhost:11434"

settings = Settings()
