"""
Configuration management for the CSV Analysis Platform.

This module handles environment variables, application settings, and provides
type-safe configuration access throughout the application.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="sqlite:///./data.db",
        description="Database connection URL"
    )
    
    # File Upload Configuration
    UPLOAD_DIR: str = Field(
        default="./uploads",
        description="Directory for uploaded files"
    )
    MAX_FILE_SIZE: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="Maximum file size in bytes"
    )
    ALLOWED_EXTENSIONS: set = Field(
        default={".csv"},
        description="Allowed file extensions"
    )
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM integration"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4",
        description="OpenAI model to use for analysis"
    )
    OPENAI_MAX_TOKENS: int = Field(
        default=4000,
        description="Maximum tokens for LLM responses"
    )
    
    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for job queue"
    )
    
    # Application Configuration
    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for session management"
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="Requests per minute per IP"
    )
    
    # Data Processing Configuration
    MAX_ROWS_FOR_LLM: int = Field(
        default=1000,
        description="Maximum rows to send to LLM for analysis"
    )
    SAMPLE_SIZE: int = Field(
        default=100,
        description="Number of rows to sample for LLM context"
    )
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = Field(
        default=30,
        description="WebSocket heartbeat interval in seconds"
    )
    
    # Test Mode
    TEST_MODE: bool = Field(
        default=False,
        description="Enable test mode (disables certain validations)"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def validate_configuration() -> None:
    """Validate that all required configuration is present."""
    errors = []
    
    # Check required environment variables (skip in test mode)
    if not settings.TEST_MODE and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required (set TEST_MODE=true to skip)")
    
    # Check file system permissions
    if not os.path.exists(settings.UPLOAD_DIR):
        try:
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        except OSError as e:
            errors.append(f"Cannot create upload directory: {e}")
    
    # Check database URL format
    if not settings.DATABASE_URL.startswith(("sqlite://", "postgresql://", "mysql://")):
        errors.append("Invalid DATABASE_URL format")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")


# Validate configuration on import
if __name__ != "__main__":
    validate_configuration()
