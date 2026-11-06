import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Resume Job Discovery Agent API"
    DEBUG: bool = False
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Model Settings (Gemini)
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.1

    # Search Provider Settings
    TAVILY_API_KEY: str = ""
    MAX_SEARCH_RESULTS_PER_QUERY: int = 3

    # File Upload Limits
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def sync_google_api_keys(self):
        """Ensures both GEMINI_API_KEY and GOOGLE_API_KEY are synchronized in os.environ."""
        key = self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key
        if self.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = self.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")


settings = Settings()
settings.sync_google_api_keys()