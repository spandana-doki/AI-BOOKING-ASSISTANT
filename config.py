"""
Application-wide configuration constants.

Note: This module intentionally avoids importing Streamlit
to keep configuration decoupled from the UI layer.
"""

# App metadata
APP_NAME: str = "AI Booking Assistant"

# Gemini / model settings
# Use models that are available on the free tier.
GEMINI_MODEL: str = "gemini-1.0-pro"
GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"

# Local embedding fallback (used if Gemini embedding quota/rate limits are hit)
LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Database settings
SQLITE_DB_PATH: str = "db/booking_app.db"

# Email / SMTP settings (placeholders)
SMTP_HOST: str = "smtp.example.com"
SMTP_PORT: int = 587
SMTP_USERNAME: str = "your_username"
SMTP_PASSWORD: str = "your_password"
SMTP_USE_TLS: bool = True
SMTP_FROM_EMAIL: str = "no-reply@example.com"

# RAG chunking configuration
RAG_CHUNK_SIZE: int = 512
RAG_CHUNK_OVERLAP: int = 128



