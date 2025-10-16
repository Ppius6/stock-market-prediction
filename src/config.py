import os
from pathlib import Path
from urllib.parse import quote_plus

import pytz
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    # Database configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    # Timezone configuration
    DB_TIMEZONE = pytz.timezone(os.getenv("DB_TIMEZONE", "UTC"))
    APP_TIMEZONE = pytz.timezone(os.getenv("APP_TIMEZONE", "UTC"))
    MARKET_TIMEZONE = pytz.timezone(os.getenv("MARKET_TIMEZONE", "America/New_York"))

    # MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_BACKEND_STORE_URI = os.getenv("MLFLOW_BACKEND_STORE_URI")
    MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlflow/artifacts")

    # Model configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "stock-price-predictor")
    PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", 1))
    RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", 168))

    # API configuration
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_HOST = os.getenv("API_HOST", "0.0.0.0")

    # Data configuration
    USE_LOCAL_DATA = os.getenv("USE_LOCAL_DATA", "true").lower() == "true"
    HISTORICAL_DAYS = int(os.getenv("HISTORICAL_DAYS", 365))
    DATA_UPDATE_INTERVAL_HOURS = int(os.getenv("DATA_UPDATE_INTERVAL_HOURS", 24))

    @property
    def database_url(self):
        # URL-encode password to handle special characters like @
        encoded_password = quote_plus(self.DB_PASSWORD) if self.DB_PASSWORD else ""
        return f"postgresql://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def data_mode(self):
        return "LOCAL" if self.USE_LOCAL_DATA else "LIVE"


config = Config()
