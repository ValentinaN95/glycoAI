"""
GlucoBuddy Configuration
Contains API keys, settings, and configuration for the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dexcom API Configuration
DEXCOM_CONFIG = {
    "CLIENT_ID": os.getenv("DEXCOM_CLIENT_ID", "your_client_id_here"),
    "CLIENT_SECRET": os.getenv("DEXCOM_CLIENT_SECRET", "your_client_secret_here"),
    "REDIRECT_URI": os.getenv("DEXCOM_REDIRECT_URI", "http://localhost:8080/callback"),
    "SANDBOX_BASE_URL": "https://sandbox-api.dexcom.com",
    "PRODUCTION_BASE_URL": "https://api.dexcom.com"
}

# Claude MCP Configuration
CLAUDE_CONFIG = {
    "API_KEY": os.getenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here"),
    "MODEL": "claude-3-sonnet-20241022",
    "MAX_TOKENS": 4000,
    "TEMPERATURE": 0.7
}

# Application Settings
APP_CONFIG = {
    "APP_NAME": "GlucoBuddy",
    "VERSION": "1.0.0",
    "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
    "HOST": os.getenv("HOST", "127.0.0.1"),
    "PORT": int(os.getenv("PORT", "7860")),
    "TIMEZONE": "UTC"
}

# Glucose Analysis Settings  
GLUCOSE_CONFIG = {
    "TARGET_RANGE_LOW": 70,  # mg/dL
    "TARGET_RANGE_HIGH": 180,  # mg/dL
    "SEVERE_HYPOGLYCEMIA": 54,  # mg/dL
    "SEVERE_HYPERGLYCEMIA": 250,  # mg/dL
    "DEFAULT_DATA_DAYS": 7,  # Number of days to fetch by default
    "MIN_READINGS_FOR_ANALYSIS": 10
}

# Demo Users Configuration (Updated with more realistic profiles)
DEMO_USERS_CONFIG = {
    "sarah_g7": {
        "name": "Sarah Thompson",
        "age": 32,
        "device_type": "G7 Mobile App",
        "username": "User7",
        "description": "Active professional with Type 1 diabetes, uses G7 CGM with smartphone integration",
        "diabetes_type": "Type 1",
        "years_with_diabetes": 8,
        "typical_glucose_pattern": "stable_with_meal_spikes"
    },
    "marcus_one": {
        "name": "Marcus Rodriguez", 
        "age": 45,
        "device_type": "ONE+ Mobile App",
        "username": "User8", 
        "description": "Father of two with Type 2 diabetes, manages with Dexcom ONE+ and lifestyle changes",
        "diabetes_type": "Type 2",
        "years_with_diabetes": 3,
        "typical_glucose_pattern": "moderate_variability"
    },
    "jennifer_g6": {
        "name": "Jennifer Chen",
        "age": 28,
        "device_type": "G6 Mobile App", 
        "username": "User6",
        "description": "Graduate student with Type 1 diabetes, tech-savvy G6 user with active lifestyle",
        "diabetes_type": "Type 1", 
        "years_with_diabetes": 12,
        "typical_glucose_pattern": "exercise_related_lows"
    },
    "robert_receiver": {
        "name": "Robert Williams",
        "age": 67,
        "device_type": "G6 Touchscreen Receiver",
        "username": "User4",
        "description": "Retired teacher with Type 2 diabetes, prefers dedicated receiver device",
        "diabetes_type": "Type 2",
        "years_with_diabetes": 15, 
        "typical_glucose_pattern": "dawn_phenomenon"
    }
}

# AI Insights Configuration
AI_INSIGHTS_CONFIG = {
    "SYSTEM_PROMPT": """You are GlucoBuddy, an AI assistant specialized in analyzing continuous glucose monitoring (CGM) data. 
    You provide personalized, actionable insights while being supportive and encouraging. 
    Always remind users to consult with their healthcare providers for medical decisions.
    
    Focus on:
    - Pattern recognition in glucose data
    - Time-in-range analysis
    - Practical lifestyle recommendations
    - Risk identification (hypo/hyperglycemia)
    - Device-specific tips
    
    Be empathetic, clear, and evidence-based in your responses.""",
    
    "INSIGHT_CATEGORIES": [
        "glucose_stability",
        "time_in_range", 
        "hypoglycemia_risk",
        "hyperglycemia_patterns",
        "daily_patterns",
        "device_optimization",
        "lifestyle_recommendations"
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    "LOG_FILE": "glucobuddy.log",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

def validate_config():
    """Validate that required configuration is present"""
    errors = []
    
    # Check Dexcom credentials
    if DEXCOM_CONFIG["CLIENT_ID"] == "your_client_id_here":
        errors.append("Dexcom CLIENT_ID not configured")
    
    if DEXCOM_CONFIG["CLIENT_SECRET"] == "your_client_secret_here":
        errors.append("Dexcom CLIENT_SECRET not configured")
    
    # Check Claude API key
    if CLAUDE_CONFIG["API_KEY"] == "your_anthropic_api_key_here":
        errors.append("Anthropic API_KEY not configured")
    
    if errors:
        print("⚠️  Configuration warnings:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 For demo purposes, the app will use simulated data.")
        print("   Create a .env file with your actual credentials for production use.")
    
    return len(errors) == 0

if __name__ == "__main__":
    validate_config()