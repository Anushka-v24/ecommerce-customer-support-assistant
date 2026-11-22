"""Environment-based settings for the application."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    APP_ENV = os.getenv("APP_ENV", "development")
    APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT = int(os.getenv("APP_PORT", "5000"))
    APP_DEBUG = _as_bool(os.getenv("APP_DEBUG"), default=False)
    SECRET_KEY = os.getenv("SECRET_KEY")
    CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    MAX_CHAT_MESSAGE_LENGTH = int(os.getenv("MAX_CHAT_MESSAGE_LENGTH", "10000"))
    MAX_SESSION_ID_LENGTH = int(os.getenv("MAX_SESSION_ID_LENGTH", "128"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(64 * 1024)))
    # Per-turn monitoring records and thumbs up/down feedback (JSONL, Git-ignored).
    TURN_LOG_ENABLED = _as_bool(os.getenv("TURN_LOG_ENABLED"), default=True)
    TURN_LOG_DIR = os.getenv("TURN_LOG_DIR", str(PROJECT_ROOT / "data" / "runtime"))
    MAX_FEEDBACK_COMMENT_LENGTH = int(os.getenv("MAX_FEEDBACK_COMMENT_LENGTH", "1000"))
    # Optional read-only FakeStore catalog. No inventory quantities are exposed by the API.
    FAKESTORE_ENABLED = _as_bool(os.getenv("FAKESTORE_ENABLED"), default=False)
    FAKESTORE_API_BASE_URL = os.getenv("FAKESTORE_API_BASE_URL", "https://fakestoreapi.com").rstrip("/")
    FAKESTORE_TIMEOUT_SECONDS = float(os.getenv("FAKESTORE_TIMEOUT_SECONDS", "4"))

    # Optional until a repository or checkpoint store explicitly requests DB use.
    DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or None
    # JSON remains the default until an operator explicitly opts into PostgreSQL.
    BUSINESS_DATA_BACKEND = os.getenv("BUSINESS_DATA_BACKEND", "json").strip().lower()
    CHECKPOINT_BACKEND = os.getenv("CHECKPOINT_BACKEND", "memory").strip().lower()
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "5"))
    DB_POOL_TIMEOUT_SECONDS = int(os.getenv("DB_POOL_TIMEOUT_SECONDS", "30"))
    DB_POOL_RECYCLE_SECONDS = int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))
    DB_CONNECT_TIMEOUT_SECONDS = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))

    # The classifier bundle is shipped with this project. An optional relative
    # override remains available for controlled local deployments.
    XLMR_MODEL_PATH = os.getenv(
        "XLMR_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "xlmr_intent_classifier"),
    )
    XLMR_LABEL_MAPPING_PATH = os.getenv(
        "XLMR_LABEL_MAPPING_PATH",
        str(Path(XLMR_MODEL_PATH) / "label_mapping.json"),
    )
    XLMR_INTENT_DESCRIPTIONS_PATH = os.getenv(
        "XLMR_INTENT_DESCRIPTIONS_PATH",
        str(PROJECT_ROOT / "app" / "classifier" / "intent_descriptions.json"),
    )
    INTENT_CONFIDENCE_THRESHOLD = float(
        os.getenv("INTENT_CONFIDENCE_THRESHOLD", "95.0")
    )
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    # When Groq is unavailable (outage or quota), accept the classifier at or
    # above this confidence instead of refusing; set above 100 to always refuse.
    VERIFIER_FALLBACK_MIN_CONFIDENCE = float(os.getenv("VERIFIER_FALLBACK_MIN_CONFIDENCE", "80.0"))
    # Structured Groq fallback for issue types and product names the rules miss.
    LLM_EXTRACTION_ENABLED = _as_bool(os.getenv("LLM_EXTRACTION_ENABLED"), default=True)
    LLM_EXTRACTION_TIMEOUT_SECONDS = float(os.getenv("LLM_EXTRACTION_TIMEOUT_SECONDS", "8"))
    # Reply in the customer's script (Hindi/Hinglish) by translating the English reply with Groq.
    REPLY_TRANSLATION_ENABLED = _as_bool(os.getenv("REPLY_TRANSLATION_ENABLED"), default=True)
    REPLY_TRANSLATION_TIMEOUT_SECONDS = float(os.getenv("REPLY_TRANSLATION_TIMEOUT_SECONDS", "6"))
    # Hand frustrated customers to a person (frustration score over the last 3 turns).
    ESCALATION_ENABLED = _as_bool(os.getenv("ESCALATION_ENABLED"), default=True)
    # Damage-photo evidence. Llama 4 Scout is not on every Groq account; any Groq vision model works.
    VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
    # Two requests in one message ("track ORD-1001 and cancel ORD-1002").
    MULTI_INTENT_ENABLED = _as_bool(os.getenv("MULTI_INTENT_ENABLED"), default=True)
    MULTI_INTENT_MIN_CONFIDENCE = float(os.getenv("MULTI_INTENT_MIN_CONFIDENCE", "90"))
    EVIDENCE_MAX_BYTES = int(os.getenv("EVIDENCE_MAX_BYTES", str(4 * 1024 * 1024)))
    EVIDENCE_TIMEOUT_SECONDS = float(os.getenv("EVIDENCE_TIMEOUT_SECONDS", "20"))
    # Ask the customer to confirm the understood request before ticket-raising workflows run.
    INTENT_CONFIRMATION_ENABLED = _as_bool(os.getenv("INTENT_CONFIRMATION_ENABLED"), default=True)

    # RAG providers are configured separately and instantiated only on demand.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip() or None
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip() or None
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
    # Free by default: a local embedding model, a local vector file and Groq for answers.
    # Set these to openai / pinecone / deepseek to use the paid providers instead.
    RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "local").strip().lower()
    RAG_VECTOR_STORE = os.getenv("RAG_VECTOR_STORE", "local").strip().lower()
    RAG_GENERATION_PROVIDER = os.getenv("RAG_GENERATION_PROVIDER", "groq").strip().lower()
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3").strip()
    LOCAL_VECTOR_STORE_PATH = os.getenv("LOCAL_VECTOR_STORE_PATH", str(PROJECT_ROOT / "data" / "runtime" / "vector_store.json"))
    RAG_GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL", "").strip() or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
    RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    RAG_KNOWLEDGE_DIR = os.getenv("RAG_KNOWLEDGE_DIR", str(PROJECT_ROOT / "knowledge_base" / "approved"))
    RAG_DEMO_KNOWLEDGE_DIR = os.getenv("RAG_DEMO_KNOWLEDGE_DIR", str(PROJECT_ROOT / "knowledge_base" / "demo"))
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ecom-support-kb-approved-v1").strip()
    PINECONE_DEMO_NAMESPACE = os.getenv("PINECONE_DEMO_NAMESPACE", "ecom-support-kb-demo-v1").strip()
    RAG_USE_DEMO_KNOWLEDGE = os.getenv("RAG_USE_DEMO_KNOWLEDGE", "false").strip().lower() in {"1", "true", "yes"}
