"""
GlucoBuddy Configuration File
This file contains all configuration settings for the GlucoBuddy application,
including Dexcom API credentials, Claude MCP settings, and application preferences.

IMPORTANT: 
- Replace placeholder values with your actual credentials
- Never commit this file to version control with real credentials
- Consider using environment variables for sensitive data in production
"""

import os
from typing import Dict, Any

# =============================================================================
# DEXCOM API CONFIGURATION
# =============================================================================

# Dexcom Developer Portal Credentials
# Get these from: https://developer.dexcom.com/
DEXCOM_CONFIG = {
    # REQUIRED: Replace with your actual Dexcom Developer Portal credentials
    "CLIENT_ID": os.getenv("DEXCOM_CLIENT_ID", "your_client_id_here"),
    "CLIENT_SECRET": os.getenv("DEXCOM_CLIENT_SECRET", "your_client_secret_here"),
    
    # OAuth Redirect URI - must match exactly what you registered in Developer Portal
    "REDIRECT_URI": "http://localhost:8080/callback",
    
    # API Endpoints
    "SANDBOX_BASE_URL": "https://sandbox-api.dexcom.com",
    "PRODUCTION_BASE_URL": "https://api.dexcom.com",
    
    # Environment setting - set to False for production
    "USE_SANDBOX": True,
    
    # OAuth Scopes
    "SCOPES": ["offline_access"],
    
    # Token refresh settings
    "TOKEN_REFRESH_BUFFER_MINUTES": 5,  # Refresh token 5 minutes before expiration
    "MAX_RETRY_ATTEMPTS": 3,
}

# =============================================================================
# CLAUDE MCP CONFIGURATION
# =============================================================================

# Claude API Configuration
# Get your API key from: https://console.anthropic.com/
CLAUDE_CONFIG = {
    # REQUIRED: Replace with your actual Anthropic API key
    "API_KEY": os.getenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here"),
    
    # Model selection
    "MODEL": "claude-3-5-sonnet-20241022",  # Updated to latest model
    
    # Generation parameters
    "MAX_TOKENS": 4000,
    "TEMPERATURE": 0.3,  # Lower for more consistent medical advice
    
    # Timeout settings
    "REQUEST_TIMEOUT": 60,  # seconds
    "MAX_RETRIES": 2,
    
    # Content filtering
    "ENABLE_SAFETY_FILTERS": True,
}

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Logging Configuration
LOGGING_CONFIG = {
    "LEVEL": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_FILE": "glucobuddy.log",
    "MAX_LOG_SIZE_MB": 10,
    "BACKUP_COUNT": 3,
    "ENABLE_CONSOLE_LOGGING": True,
    "ENABLE_FILE_LOGGING": True,
}

# Data Processing Settings
DATA_CONFIG = {
    # Default data range for analysis
    "DEFAULT_DAYS_BACK": 7,
    "MAX_DAYS_BACK": 30,
    
    # Glucose thresholds (mg/dL)
    "TARGET_RANGE_LOW": 70,
    "TARGET_RANGE_HIGH": 180,
    "HYPOGLYCEMIA_THRESHOLD": 70,
    "HYPERGLYCEMIA_THRESHOLD": 250,
    
    # Time in Range targets (percentages)
    "EXCELLENT_TIR_THRESHOLD": 80,
    "GOOD_TIR_THRESHOLD": 70,
    "ACCEPTABLE_TIR_THRESHOLD": 50,
    
    # Data quality settings
    "MIN_READINGS_FOR_ANALYSIS": 20,
    "MAX_GAP_HOURS": 3,  # Maximum gap between readings for continuous analysis
    
    # Pattern detection settings
    "PATTERN_DETECTION_MIN_DAYS": 3,
    "SIGNIFICANT_TREND_THRESHOLD": 2.0,  # mg/dL per reading
}

# =============================================================================
# DEMO USER CONFIGURATION
# =============================================================================

# Demo mode settings - for development and testing
DEMO_CONFIG = {
    "ENABLE_DEMO_MODE": True,
    "DEFAULT_DEMO_USER": "sarah_g7",  # Which demo user to use by default
    "GENERATE_SYNTHETIC_DATA": True,  # Generate realistic demo data when API fails
    "DEMO_DATA_DAYS": 14,  # Days of demo data to generate
    "DEMO_DATA_NOISE_FACTOR": 0.1,  # Randomness in demo data (0.0 to 1.0)
}

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

SECURITY_CONFIG = {
    # Data encryption (for local storage)
    "ENCRYPT_LOCAL_DATA": True,
    "ENCRYPTION_KEY": os.getenv("GLUCOBUDDY_ENCRYPTION_KEY", None),
    
    # Session management
    "SESSION_TIMEOUT_MINUTES": 30,
    "REQUIRE_REAUTHENTICATION": True,
    
    # Data retention
    "MAX_LOCAL_DATA_DAYS": 90,
    "AUTO_CLEANUP_OLD_DATA": True,
    
    # Privacy settings
    "ANONYMIZE_LOGS": True,
    "ALLOW_TELEMETRY": False,
}

# =============================================================================
# UI/UX CONFIGURATION
# =============================================================================

UI_CONFIG = {
    # Display preferences
    "DEFAULT_GLUCOSE_UNIT": "mg/dL",  # or "mmol/L"
    "DATE_FORMAT": "%Y-%m-%d %H:%M",
    "TIMEZONE": "local",  # or specific timezone like "US/Eastern"
    
    # Chart settings
    "CHART_HEIGHT": 400,
    "CHART_WIDTH": 800,
    "SHOW_TARGET_RANGE": True,
    "ENABLE_INTERACTIVE_CHARTS": True,
    
    # Notification settings
    "ENABLE_DESKTOP_NOTIFICATIONS": True,
    "ALERT_SOUND": True,
    "NOTIFICATION_TIMEOUT": 5,  # seconds
}

# =============================================================================
# INTEGRATION SETTINGS
# =============================================================================

# External service integrations
INTEGRATION_CONFIG = {
    # Health app integrations
    "ENABLE_APPLE_HEALTH": False,
    "ENABLE_GOOGLE_FIT": False,
    
    # Export formats
    "SUPPORTED_EXPORT_FORMATS": ["CSV", "JSON", "PDF"],
    "DEFAULT_EXPORT_FORMAT": "CSV",
    
    # Webhook settings (for advanced users)
    "WEBHOOK_URL": None,
    "WEBHOOK_SECRET": os.getenv("GLUCOBUDDY_WEBHOOK_SECRET", None),
    "ENABLE_WEBHOOKS": False,
}

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Development and debugging options
DEV_CONFIG = {
    "DEBUG_MODE": False,
    "ENABLE_API_MOCKING": False,  # Mock API calls for development
    "VERBOSE_LOGGING": False,
    "SAVE_RAW_API_RESPONSES": False,
    "ENABLE_PERFORMANCE_MONITORING": False,
    
    # Testing settings
    "RUN_TESTS_ON_STARTUP": False,
    "TEST_DATA_PATH": "test_data/",
}

# =============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# =============================================================================

# Override settings based on environment
ENVIRONMENT = os.getenv("GLUCOBUDDY_ENV", "development").lower()

if ENVIRONMENT == "production":
    # Production overrides
    DEXCOM_CONFIG["USE_SANDBOX"] = False
    LOGGING_CONFIG["LEVEL"] = "WARNING"
    DEMO_CONFIG["ENABLE_DEMO_MODE"] = False
    DEV_CONFIG["DEBUG_MODE"] = False
    SECURITY_CONFIG["REQUIRE_REAUTHENTICATION"] = True
    
elif ENVIRONMENT == "testing":
    # Testing overrides
    LOGGING_CONFIG["LEVEL"] = "DEBUG"
    DEV_CONFIG["ENABLE_API_MOCKING"] = True
    DEV_CONFIG["RUN_TESTS_ON_STARTUP"] = True
    DEMO_CONFIG["ENABLE_DEMO_MODE"] = True

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config() -> Dict[str, Any]:
    """
    Validate configuration settings and return validation results.
    Returns a dictionary with validation status and any errors.
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required Dexcom credentials
    if DEXCOM_CONFIG["CLIENT_ID"] == "your_client_id_here":
        validation_results["errors"].append("Dexcom CLIENT_ID not configured")
        validation_results["valid"] = False
    
    if DEXCOM_CONFIG["CLIENT_SECRET"] == "your_client_secret_here":
        validation_results["errors"].append("Dexcom CLIENT_SECRET not configured")
        validation_results["valid"] = False
    
    # Check Claude API key
    if CLAUDE_CONFIG["API_KEY"] == "your_anthropic_api_key_here":
        validation_results["warnings"].append("Claude API key not configured - AI insights will be limited")
    
    # Validate glucose thresholds
    if DATA_CONFIG["TARGET_RANGE_LOW"] >= DATA_CONFIG["TARGET_RANGE_HIGH"]:
        validation_results["errors"].append("Invalid glucose target range")
        validation_results["valid"] = False
    
    # Check encryption key for production
    if ENVIRONMENT == "production" and SECURITY_CONFIG["ENCRYPT_LOCAL_DATA"]:
        if not SECURITY_CONFIG["ENCRYPTION_KEY"]:
            validation_results["warnings"].append("No encryption key set for production environment")
    
    return validation_results

def get_active_config() -> Dict[str, Any]:
    """
    Get the complete active configuration as a dictionary.
    Useful for debugging and configuration inspection.
    """
    return {
        "dexcom": DEXCOM_CONFIG,
        "claude": CLAUDE_CONFIG,
        "logging": LOGGING_CONFIG,
        "data": DATA_CONFIG,
        "demo": DEMO_CONFIG,
        "security": SECURITY_CONFIG,
        "ui": UI_CONFIG,
        "integration": INTEGRATION_CONFIG,
        "development": DEV_CONFIG,
        "environment": ENVIRONMENT
    }

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_dexcom_base_url() -> str:
    """Get the appropriate Dexcom API base URL based on environment."""
    return (DEXCOM_CONFIG["SANDBOX_BASE_URL"] if DEXCOM_CONFIG["USE_SANDBOX"] 
            else DEXCOM_CONFIG["PRODUCTION_BASE_URL"])

def is_demo_mode_enabled() -> bool:
    """Check if demo mode is enabled."""
    return DEMO_CONFIG["ENABLE_DEMO_MODE"]

def get_glucose_unit() -> str:
    """Get the configured glucose unit."""
    return UI_CONFIG["DEFAULT_GLUCOSE_UNIT"]

# =============================================================================
# STARTUP CONFIGURATION CHECK
# =============================================================================

if __name__ == "__main__":
    print("GlucoBuddy Configuration Validation")
    print("=" * 40)
    
    results = validate_config()
    
    if results["valid"]:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    if results["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    print(f"\nEnvironment: {ENVIRONMENT}")
    print(f"Demo Mode: {'Enabled' if is_demo_mode_enabled() else 'Disabled'}")
    print(f"Sandbox Mode: {'Enabled' if DEXCOM_CONFIG['USE_SANDBOX'] else 'Disabled'}")
    print(f"Claude Integration: {'Enabled' if CLAUDE_CONFIG['API_KEY'] != 'your_anthropic_api_key_here' else 'Disabled'}")