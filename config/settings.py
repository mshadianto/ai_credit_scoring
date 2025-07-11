"""
Configuration settings for AI Credit Scoring System
FIXED VERSION - No Pydantic Dependencies Required
"""
import os
from pathlib import Path
from typing import Dict, List

# Base directory
BASE_DIR = Path(__file__).parent.parent

class Settings:
    """Application settings - Simple version without external dependencies"""
    
    def __init__(self):
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # Continue without dotenv if not available
        
        # App Configuration
        self.APP_NAME = "AI Credit Scoring System"
        self.APP_VERSION = "1.0.0"
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        
        # API Configuration
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
        self.DEFAULT_MODEL = "qwen/qwen-2.5-7b-instruct"
        
        # Database Configuration
        self.DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/credit_scoring.db")
        self.DATABASE_ECHO = os.getenv("DATABASE_ECHO", "False").lower() == "true"
        
        # ML Model Configuration
        self.MODEL_DIR = BASE_DIR / "models" / "trained"
        self.DATA_DIR = BASE_DIR / "data"
        self.SAMPLE_DATA_SIZE = 10000
        
        # Credit Scoring Parameters
        self.CREDIT_SCORE_RANGE = (300, 850)
        self.RISK_THRESHOLDS = {
            "very_low": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8
        }
        
        # Credit Types Configuration
        self.CREDIT_TYPES = [
            "Personal Loan",
            "Mortgage", 
            "Credit Card",
            "Auto Loan",
            "Business Loan"
        ]
        
        # Interest Rate Ranges
        self.INTEREST_RATES = {
            "excellent": {"min": "5.5%", "max": "7.0%"},
            "good": {"min": "7.0%", "max": "9.5%"},
            "fair": {"min": "9.5%", "max": "15.0%"},
            "poor": {"min": "15.0%", "max": "25.0%"}
        }
        
        # Model Training Configuration
        self.MODEL_PARAMS = {
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            }
        }
        
        # Feature Configuration
        self.NUMERICAL_FEATURES = [
            "age", "income", "employment_years", "debt_to_income",
            "credit_history_length", "number_of_accounts", 
            "payment_history", "credit_utilization"
        ]
        
        self.CATEGORICAL_FEATURES = [
            "education_level", "marital_status", "home_ownership"
        ]
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = BASE_DIR / "logs" / "credit_scoring.log"
        
        # Security
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 30
        
        # Performance
        try:
            self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
        except:
            self.MAX_WORKERS = 4
        self.CACHE_TTL = 3600  # 1 hour
        
        # Streamlit Configuration
        self.STREAMLIT_CONFIG = {
            "page_title": "AI Credit Scoring",
            "page_icon": "🏦",
            "layout": "wide",
            "initial_sidebar_state": "expanded"
        }

# Global settings instance
settings = Settings()

# Banking Industry Standards
BANKING_STANDARDS = {
    "basel_iii": {
        "minimum_capital_ratio": 0.08,
        "leverage_ratio": 0.03,
        "liquidity_coverage_ratio": 1.0
    },
    "indonesia_regulations": {
        "bi_regulation": "PBI No. 23/6/PBI/2021",
        "ojk_regulation": "POJK No. 40/POJK.03/2019",
        "max_ltv_mortgage": 0.9,
        "max_dti_ratio": 0.4
    }
}

# Risk Assessment Matrices
RISK_MATRICES = {
    "default_probability": {
        "very_low": (0.0, 0.02),
        "low": (0.02, 0.05),
        "medium": (0.05, 0.10),
        "high": (0.10, 0.20),
        "very_high": (0.20, 1.0)
    },
    "credit_score_ranges": {
        "excellent": (740, 850),
        "good": (670, 739),
        "fair": (580, 669),
        "poor": (300, 579)
    }
}

def get_db_url() -> str:
    """Get database URL with proper path handling"""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        # Ensure directory exists for SQLite
        db_path = Path(db_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_url

def create_directories():
    """Create necessary directories"""
    directories = [
        settings.MODEL_DIR,
        settings.DATA_DIR / "raw",
        settings.DATA_DIR / "processed", 
        settings.DATA_DIR / "sample",
        BASE_DIR / "logs",
        BASE_DIR / "exports"
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {directory}: {e}")

# Initialize directories on import
create_directories()