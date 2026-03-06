# app/config.py

from pydantic_settings import BaseSettings
from pydantic import Field
from loguru import logger
import sys

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Brain Tumor Detection API"
    app_version: str = "1.0.0"
    app_env: str = Field(default="development", env="APP_ENV")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")


    # Model settings
    model_path: str = Field(default="app/ml_model/best.pth", env="MODEL_PATH")
    model_name: str = Field(default="EfficientNetV2-S-BrainTumor", env="MODEL_NAME")
    model_version: str = Field(default="1.0.0", env="MODEL_VERSION")

    # Authentication settings
    api_keys: str = Field(
        default="sk-tumor-dev-abc123, sk-tumor-prod-xyz789", 
        env="API_KEYS",
        description="Comma-separated list of valid API keys for authentication"
    )

    # Inference settings
    max_image_mb: float = Field(default=5.0, env="MAX_IMAGE_MB", description="Maximum allowed image size in megabytes")
    max_batch_size: int = Field(default=8, env="MAX_BATCH_SIZE", description="Maximum batch size for inference")

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL", description="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_api_keys(self):
        return [key.strip() for key in self.api_keys.split(",")]
    

# Logging configuration
def configure_logging(log_level: str = "INFO"):
    logger.remove()  # Remove default logger
    logger.add(
        sys.stdout, 
        level=log_level.upper(), 
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    )

    logger.add(
        "logs/api.log",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )


# Alias for main.py
setup_logging = configure_logging

# Initialize settings
settings = Settings()
    