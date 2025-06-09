#!/usr/bin/env python3
"""
Secure Configuration Management for GlycoAI
Handles API keys and secrets safely for both local development and Hugging Face Spaces
"""

import os
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration manager for API keys and secrets"""
    
    def __init__(self):
        self.config = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from environment variables"""
        
        # Mistral AI Configuration
        self.mistral_api_key = self._get_secret(
            "MISTRAL_API_KEY",
            description="Mistral AI API Key"
        )
        
        self.mistral_agent_id = self._get_secret(
            "MISTRAL_AGENT_ID", 
            description="Mistral AI Agent ID",
            required=False
        )
        
        # Dexcom API Configuration (for future real API integration)
        self.dexcom_client_id = self._get_secret(
            "DEXCOM_CLIENT_ID",
            description="Dexcom API Client ID",
            required=False
        )
        
        self.dexcom_client_secret = self._get_secret(
            "DEXCOM_CLIENT_SECRET",
            description="Dexcom API Client Secret",
            required=False
        )
        
        # Application Configuration
        self.app_environment = os.getenv("ENVIRONMENT", "development")
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Hugging Face Space Detection
        self.is_huggingface_space = os.getenv("SPACE_ID") is not None
        
        logger.info(f"Configuration loaded for environment: {self.app_environment}")
        logger.info(f"Running on Hugging Face Space: {self.is_huggingface_space}")
    
    def _get_secret(self, key: str, description: str = "", required: bool = True) -> Optional[str]:
        """Safely get secret from environment variables"""
        value = os.getenv(key)
        
        if value:
            logger.info(f"✅ {description or key} loaded successfully")
            return value
        elif required:
            logger.error(f"❌ Required secret {key} ({description}) not found!")
            logger.error(f"Please set the {key} environment variable")
            return None
        else:
            logger.warning(f"⚠️  Optional secret {key} ({description}) not set")
            return None
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate that all required configuration is present"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required secrets
        if not self.mistral_api_key:
            validation_result["valid"] = False
            validation_result["errors"].append("MISTRAL_API_KEY is required")
        
        # Check optional but recommended secrets
        if not self.mistral_agent_id:
            validation_result["warnings"].append("MISTRAL_AGENT_ID not set - will use standard chat completion")
        
        # Environment-specific checks
        if self.is_huggingface_space:
            if not all([self.mistral_api_key]):
                validation_result["errors"].append("Hugging Face Space requires MISTRAL_API_KEY in secrets")
        
        return validation_result
    
    def get_mistral_config(self) -> Dict[str, Optional[str]]:
        """Get Mistral AI configuration"""
        return {
            "api_key": self.mistral_api_key,
            "agent_id": self.mistral_agent_id
        }
    
    def get_dexcom_config(self) -> Dict[str, Optional[str]]:
        """Get Dexcom API configuration"""
        return {
            "client_id": self.dexcom_client_id,
            "client_secret": self.dexcom_client_secret
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.app_environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.app_environment == "production"

# Global configuration instance
config = SecureConfig()

def get_config() -> SecureConfig:
    """Get the global configuration instance"""
    return config

def validate_environment():
    """Validate environment configuration and provide helpful messages"""
    print("🔍 Validating GlycoAI Configuration...")
    print("=" * 50)
    
    validation = config.validate_configuration()
    
    if validation["valid"]:
        print("✅ Configuration validation passed!")
    else:
        print("❌ Configuration validation failed!")
        for error in validation["errors"]:
            print(f"   ❌ {error}")
    
    if validation["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"   ⚠️  {warning}")
    
    # Provide setup instructions
    if not validation["valid"]:
        print("\n📋 Setup Instructions:")
        print("=" * 30)
        
        if config.is_huggingface_space:
            print("🤗 For Hugging Face Spaces:")
            print("1. Go to your Space settings")
            print("2. Add Repository secrets:")
            print("   - MISTRAL_API_KEY: your_mistral_api_key")
            print("   - MISTRAL_AGENT_ID: your_agent_id (optional)")
        else:
            print("💻 For Local Development:")
            print("1. Create a .env file in your project root:")
            print("   MISTRAL_API_KEY=your_mistral_api_key")
            print("   MISTRAL_AGENT_ID=your_agent_id")
            print("2. Or set environment variables:")
            print("   export MISTRAL_API_KEY=your_mistral_api_key")
            print("   export MISTRAL_AGENT_ID=your_agent_id")
    
    return validation["valid"]

if __name__ == "__main__":
    validate_environment()