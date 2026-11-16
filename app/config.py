import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "")  # required; no default, so no credentials live in the code
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # 768 or 1536 (pgvector HNSW max is 2000)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))  # texts per request; 400 errors split the batch automatically
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "8"))
# Passages scoring below this cosine similarity are not given to the model (0 switches the cut-off off). Measured with
# gemini-embedding-2 at 768 dims: course questions score 0.67-0.80, unrelated ones 0.54-0.58. Re-measure if the
# embedding model or dimension changes.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.60"))
PRELOAD_EMBEDDING = os.getenv("PRELOAD_EMBEDDING", "1") == "1"
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "45"))  # seconds per answer-model request
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))  # extra tries after a 429 / 5xx / connection error
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "5000"))  # also the limit for each history entry
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
# Chat memory: the client sends a running summary plus recent exchanges. The follow-up rewrite only needs the last few.
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "1000"))
REWRITE_TURNS = int(os.getenv("REWRITE_TURNS", "2"))
