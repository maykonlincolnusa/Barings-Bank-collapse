from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    env: str = os.getenv("BARINGS_ENV", "dev")
    jwt_secret: str = os.getenv("BARINGS_JWT_SECRET", "change-me-in-production")
    admin_username: str = os.getenv("BARINGS_ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("BARINGS_ADMIN_PASSWORD", "admin123")
    analyst_username: str = os.getenv("BARINGS_ANALYST_USERNAME", "analyst")
    analyst_password: str = os.getenv("BARINGS_ANALYST_PASSWORD", "analyst123")
    rate_limit_per_minute: int = int(os.getenv("BARINGS_RATE_LIMIT_PER_MINUTE", "60"))
    api_host: str = os.getenv("BARINGS_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("BARINGS_API_PORT", "8000"))
    streamlit_port: int = int(os.getenv("BARINGS_STREAMLIT_PORT", "8501"))
    api_base_url: str = os.getenv("BARINGS_API_BASE_URL", "http://localhost:8000")
    encryption_key: str = os.getenv("BARINGS_ENCRYPTION_KEY", "")


settings = Settings()

