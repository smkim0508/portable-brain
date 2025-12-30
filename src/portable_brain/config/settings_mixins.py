# mixin settings for external services like db, llm, vector, etc.
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class MainDBSettingsMixin(BaseModel):
    """
    Model for common SQLAlchemy connection pool settings.
    """
    MAIN_DB_POOL_SIZE: int = Field(default=5, description="Number of connections to keep in the pool.")
    MAIN_DB_MAX_OVERFLOW: int = Field(default=10, description="Max 'overflow' connections beyond pool_size.")
    MAIN_DB_POOL_TIMEOUT: int = Field(default=30, description="Seconds to wait before giving up on getting a connection.")
    MAIN_DB_POOL_RECYCLE: int = Field(default=1800, description="Recycle connections after this many seconds.")
    
    MAIN_DB_USER: str
    MAIN_DB_PW: str
    MAIN_DB_HOST: str
    MAIN_DB_PORT: str
    MAIN_DB_NAME: str

class GoogleGenAISettingsMixin(BaseModel):
    """
    Model for Google GenAI LLM Client settings.
    """
    GOOGLE_GENAI_API_KEY: str

    