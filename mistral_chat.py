#!/usr/bin/env python3
"""
GlucoBuddy Mistral Chat Integration - Secure Version
No hardcoded API keys - uses environment variables and Hugging Face secrets
"""

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import asdict
import requests
import random
import numpy as np
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from apifunctions import (
    DexcomAPI,
    GlucoseAnalyzer,
    DEMO_USERS,
    DemoUser,
    format_glucose_data_for_display
)

# Import secure configuration
from config import get_config, validate_environment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlucoseDataGenerator:
    """Generate realistic mock glucose data for testing and demo purposes"""
    
    @staticmethod
    def create_realistic_pattern(days: int = 14, user_type: str = "normal") -> List[Dict]:
        """Generate glucose data with realistic patterns"""
        data_points = []
        start_time = datetime.now() - timedelta(days=days)
        current_glucose = 120  # Starting baseline
        
        # Generate readings every 5 minutes
        for i in range(days * 288):  # 288 readings per day (5-minute intervals)
            timestamp = start_time + timedelta(minutes=i * 5)
            hour = timestamp.hour
            
            # Simulate daily patterns
            daily_variation = GlucoseDataGenerator._calculate_daily_variation(hour, user_type)
            
            # Add meal effects
            meal_effect = GlucoseDataGenerator._calculate_meal_effects(hour, i)
            
            # Random variation
            random_noise = random.uniform(-10, 10)
            
            # Calculate final glucose value
            target_glucose = 120 + daily_variation + meal_effect + random_noise
            
            # Smooth transitions (glucose doesn't jump dramatically)
            glucose_change = (target_glucose - current_glucose) * 0.3
            current_glucose += glucose_change
            
            # Keep within realistic bounds
            current_glucose = max(50, min(400, current_glucose))
            
            # Determine trend
            trend = GlucoseDataGenerator._calculate_trend(glucose_change)
            
            data_points.append({
                'systemTime': timestamp.isoformat(),
                'displayTime': timestamp.isoformat(),
                'value': round(current_glucose),
                'trend': trend,
                'realtimeValue': round(current_glucose),
                'smoothedValue': round(current_glucose)
            })
        
        return data_points
    
    @staticmethod
    def _calculate_daily_variation(hour: int, user_type: str) -> float:
        """Calculate glucose variation based on time of day"""
        if user_type == "dawn_phenomenon":
            # Higher glucose in early morning
            if 4 <= hour <= 8:
                return 30 + 20 * np.sin((hour - 4) * np.pi / 4)
            return 10 * np.sin((hour - 12) * np.pi / 12)
        
        elif user_type == "night_low":
            # Lower glucose at night
            if 22 <= hour or hour <= 6:
                return -20
            return 5 * np.sin((hour - 12) * np.pi / 12)
        
        else:  # Normal pattern
            return 15 * np.sin((hour - 6) * np.pi / 12)
    
    @staticmethod
    def _calculate_meal_effects(hour: int, reading_index: int) -> float:
        """Calculate glucose spikes from meals"""
        meal_times = [7, 12, 18]  # Breakfast, lunch, dinner
        meal_effect = 0
        
        for meal_time in meal_times:
            if abs(hour - meal_time) <= 2:
                # Create post-meal spike pattern
                time_since_meal = abs(hour - meal_time)
                if time_since_meal <= 1:
                    meal_effect += 40 * (1 - time_since_meal)
                else:
                    meal_effect += 20 * (2 - time_since_meal)
        
        return meal_effect
    
    @staticmethod
    def _calculate_trend(glucose_change: float) -> str:
        """Determine trend arrow based on glucose change"""
        if glucose_change > 5:
            return 'singleUp'
        elif glucose_change > 2:
            return 'fortyFiveUp'
        elif glucose_change < -5:
            return 'singleDown'
        elif glucose_change < -2:
            return 'fortyFiveDown'
        else:
            return 'flat'

class SafeDataProcessor:
    """Safe data processing utilities to avoid pandas/numpy issues"""
    
    @staticmethod
    def safe_convert_to_json(obj) -> Any:
        """Safely convert any object to JSON-serializable format"""
        try:
            if obj is None:
                return None
            
            # Handle pandas/numpy scalars
            if hasattr(obj, 'item'):
                try:
                    return obj.item()
                except (ValueError, TypeError):
                    pass
            
            # Handle numpy types
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [SafeDataProcessor.safe_convert_to_json(item) for item in obj.tolist()]
            
            # Handle pandas types
            elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return [SafeDataProcessor.safe_convert_to_json(item) for item in obj.tolist()]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            
            # Handle collections
            elif isinstance(obj, dict):
                return {key: SafeDataProcessor.safe_convert_to_json(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [SafeDataProcessor.safe_convert_to_json(item) for item in obj]
            
            # Handle datetime
            elif isinstance(obj, datetime):
                return obj.isoformat()
            
            # Check for NaN in a safe way
            elif str(obj).lower() in ['nan', 'nat', 'none']:
                return None
            
            # Try direct conversion for basic types
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            
            # Last resort - convert to string
            else:
                return str(obj)
                
        except Exception as e:
            logger.warning(f"Error converting object to JSON: {e}, object type: {type(obj)}")
            return str(obj) if obj is not None else None
    
    @staticmethod
    def safe_calculate_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Safely calculate glucose statistics without pandas boolean errors"""
        if df is None or df.empty:
            return {"error": "No data available"}
        
        try:
            # Get glucose values as numpy array
            glucose_col = df.get('value')
            if glucose_col is None:
                return {"error": "No glucose values found"}
            
            # Clean data - remove NaN values
            clean_values = glucose_col.dropna()
            if len(clean_values) == 0:
                return {"error": "No valid glucose values"}
            
            # Convert to numpy array for safe calculations
            values = np.array(clean_values.tolist(), dtype=float)
            
            # Calculate basic statistics
            avg_glucose = float(np.mean(values))
            min_glucose = float(np.min(values))
            max_glucose = float(np.max(values))
            std_glucose = float(np.std(values))
            total_readings = int(len(values))
            
            # Calculate time in ranges safely
            in_range_count = int(np.sum((values >= 70) & (values <= 180)))
            below_range_count = int(np.sum(values < 70))
            above_range_count = int(np.sum(values > 180))
            
            time_in_range = (in_range_count / total_readings) * 100 if total_readings > 0 else 0
            time_below_70 = (below_range_count / total_readings) * 100 if total_readings > 0 else 0
            time_above_180 = (above_range_count / total_readings) * 100 if total_readings > 0 else 0
            
            return {
                "average_glucose": avg_glucose,
                "min_glucose": min_glucose,
                "max_glucose": max_glucose,
                "std_glucose": std_glucose,
                "time_in_range_70_180": time_in_range,
                "time_below_70": time_below_70,
                "time_above_180": time_above_180,
                "total_readings": total_readings
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {"error": f"Statistics calculation failed: {str(e)}"}
    
    @staticmethod
    def safe_extract_recent_readings(df: pd.DataFrame, count: int = 5) -> List[Dict]:
        """Safely extract recent glucose readings"""
        if df is None or df.empty:
            return []
        
        try:
            # Get last N readings
            recent_df = df.tail(count)
            readings = []
            
            for idx, row in recent_df.iterrows():
                try:
                    # Safely extract each field
                    display_time = row.get('displayTime') or row.get('systemTime')
                    glucose_value = row.get('value')
                    trend_value = row.get('trend', 'flat')
                    
                    # Process timestamp
                    if pd.notna(display_time):
                        if isinstance(display_time, str):
                            time_str = display_time
                        else:
                            time_str = pd.to_datetime(display_time).isoformat()
                    else:
                        time_str = datetime.now().isoformat()
                    
                    # Process glucose value
                    if pd.notna(glucose_value):
                        glucose_clean = SafeDataProcessor.safe_convert_to_json(glucose_value)
                    else:
                        glucose_clean = None
                    
                    # Process trend
                    trend_clean = str(trend_value) if pd.notna(trend_value) else 'flat'
                    
                    readings.append({
                        "time": time_str,
                        "glucose": glucose_clean,
                        "trend": trend_clean
                    })
                    
                except Exception as row_error:
                    logger.warning(f"Error processing reading at index {idx}: {row_error}")
                    continue
            
            return readings
            
        except Exception as e:
            logger.error(f"Error extracting recent readings: {e}")
            return []

class MistralAPIClient:
    """Secure Mistral API client with proper configuration management"""
    
    def __init__(self, api_key: str = None, agent_id: str = None):
        # Get configuration
        config = get_config()
        
        # Use provided keys or fall back to configuration
        self.api_key = api_key or config.mistral_api_key
        self.agent_id = agent_id or config.mistral_agent_id
        
        # Validate API key
        if not self.api_key:
            raise ValueError("Mistral API key is required. Please set MISTRAL_API_KEY environment variable.")
        
        self.base_url = "https://api.mistral.ai/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        logger.info("MistralAPIClient initialized successfully")
        if self.agent_id:
            logger.info("Agent ID configured - agent completions available")
        else:
            logger.info("No agent ID - using standard chat completions only")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection with a simple request"""
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "message": "API connection successful"}
            elif response.status_code == 401:
                return {"success": False, "message": "Invalid API key"}
            elif response.status_code == 429:
                return {"success": False, "message": "Rate limit exceeded"}
            else:
                return {"success": False, "message": f"API error: {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {"success": False, "message": "Connection timeout"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"Network error: {str(e)}"}
        except Exception as e:
            return {"success": False, "message": f"Unexpected error: {str(e)}"}
    
    def chat_completion(self, messages: List[Dict], model: str = "mistral-large-latest") -> Dict[str, Any]:
        """Send chat completion request"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {})
                }
            else:
                error_detail = self._extract_error_message(response)
                return {
                    "success": False,
                    "error": f"API error {response.status_code}: {error_detail}"
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Network error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def agent_completion(self, messages: List[Dict]) -> Dict[str, Any]:
        """Send request to Mistral agent (if agent_id is available)"""
        if not self.agent_id:
            return {"success": False, "error": "No agent ID configured"}
        
        try:
            payload = {
                "agent_id": self.agent_id,
                "messages": messages,
                "max_tokens": 800
            }
            
            response = self.session.post(
                f"{self.base_url}/agents/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"]
                }
            else:
                error_detail = self._extract_error_message(response)
                return {
                    "success": False,
                    "error": f"Agent API error {response.status_code}: {error_detail}"
                }
                
        except Exception as e:
            return {"success": False, "error": f"Agent request failed: {str(e)}"}
    
    def _extract_error_message(self, response) -> str:
        """Extract error message from API response"""
        try:
            error_data = response.json()
            return error_data.get("message", error_data.get("error", "Unknown error"))
        except:
            return response.text[:200] if response.text else "Unknown error"

class GlucoBuddyMistralChat:
    """Main chat interface for glucose data analysis with Mistral AI - Secure Version"""
    
    def __init__(self, mistral_api_key: str = None, mistral_agent_id: str = None):
        # Validate environment first
        if not validate_environment():
            raise ValueError("Environment validation failed. Please check your configuration.")
        
        # Initialize components
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        self.mistral_client = MistralAPIClient(mistral_api_key, mistral_agent_id)
        self.data_processor = SafeDataProcessor()
        
        # User data
        self.current_user: Optional[DemoUser] = None
        self.current_glucose_data: Optional[pd.DataFrame] = None
        self.current_stats: Optional[Dict] = None
        self.current_patterns: Optional[Dict] = None
        
        # Chat state
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 messages
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Mistral API connection"""
        return self.mistral_client.test_connection()
    
    def load_user_data(self, user_key: str) -> Dict[str, Any]:
        """Load and process glucose data for a demo user"""
        try:
            if user_key not in DEMO_USERS:
                available_keys = ', '.join(DEMO_USERS.keys())
                return {
                    "success": False,
                    "message": f"❌ Invalid user key '{user_key}'. Available: {available_keys}"
                }
            
            self.current_user = DEMO_USERS[user_key]
            self.logger.info(f"Loading data for user: {self.current_user.name}")
            
            # Try to get real data from Dexcom API
            try:
                self.dexcom_api.simulate_demo_login(user_key)
                egv_data = self.dexcom_api.get_egv_data()
            except Exception as api_error:
                self.logger.warning(f"Dexcom API failed: {api_error}")
                egv_data = []
            
            # Fallback to generated data if no real data
            if not egv_data:
                self.logger.info("Generating mock glucose data for 14 days")
                # Determine pattern based on user
                pattern_map = {
                    "sarah_g7": "normal",
                    "marcus_one": "dawn_phenomenon", 
                    "jennifer_g6": "normal",
                    "robert_receiver": "dawn_phenomenon"
                }
                user_pattern = pattern_map.get(user_key, "normal")
                egv_data = GlucoseDataGenerator.create_realistic_pattern(days=14, user_type=user_pattern)
            
            # Process the data safely
            self.current_glucose_data = self.analyzer.process_egv_data(egv_data)
            
            if self.current_glucose_data is None or self.current_glucose_data.empty:
                return {
                    "success": False,
                    "message": "❌ Failed to process glucose data"
                }
            
            # Calculate statistics safely
            self.current_stats = self.data_processor.safe_calculate_stats(self.current_glucose_data)
            self.current_patterns = self.analyzer.identify_patterns(self.current_glucose_data)
            
            # Prepare response
            data_points = len(self.current_glucose_data)
            avg_glucose = self.current_stats.get('average_glucose', 0)
            time_in_range = self.current_stats.get('time_in_range_70_180', 0)
            
            return {
                "success": True,
                "message": f"✅ Successfully loaded data for {self.current_user.name}",
                "user": asdict(self.current_user),
                "data_points": data_points,
                "stats": self.data_processor.safe_convert_to_json(self.current_stats),
                "summary": f"📊 {data_points} readings | Avg: {avg_glucose:.1f} mg/dL | TIR: {time_in_range:.1f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load user data: {e}")
            return {
                "success": False,
                "message": f"❌ Failed to load user data: {str(e)}"
            }
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get current context for chat"""
        if not self.current_user or not self.current_stats:
            return {"error": "No user data loaded"}
        
        try:
            # Build safe context
            context = {
                "user": {
                    "name": self.current_user.name,
                    "age": self.current_user.age,
                    "diabetes_type": self.current_user.diabetes_type,
                    "device_type": self.current_user.device_type,
                    "years_with_diabetes": self.current_user.years_with_diabetes,
                    "typical_pattern": getattr(self.current_user, 'typical_glucose_pattern', 'normal')
                },
                "statistics": self.data_processor.safe_convert_to_json(self.current_stats),
                "patterns": self.data_processor.safe_convert_to_json(self.current_patterns),
                "data_points": len(self.current_glucose_data) if self.current_glucose_data is not None else 0,
                "recent_readings": self.data_processor.safe_extract_recent_readings(self.current_glucose_data)
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error building context: {e}")
            return {"error": f"Failed to build context: {str(e)}"}
    
    def build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive system prompt"""
        base_prompt = """You are GlucoBuddy, a helpful and encouraging diabetes management assistant. 

Your role:
- Provide personalized glucose management advice based on the user's actual data
- Be supportive, encouraging, and use emojis to be friendly
- Give actionable recommendations while staying within scope
- Always remind users to consult healthcare providers for medical decisions
- Reference specific data points when providing insights

Guidelines:
- Keep responses under 400 words and conversational
- Use specific numbers from the data when relevant
- Provide practical, actionable advice
- Be encouraging about progress and realistic about challenges
- Use bullet points sparingly - prefer natural conversation"""

        if context.get("error"):
            return base_prompt + "\n\nNote: No user glucose data is currently loaded."
        
        # Add user-specific context
        user_info = context.get("user", {})
        stats = context.get("statistics", {})
        
        context_addition = f"""

Current User: {user_info.get('name', 'Unknown')} ({user_info.get('age', 'N/A')} years old)
- Diabetes Type: {user_info.get('diabetes_type', 'Unknown')}
- Years with diabetes: {user_info.get('years_with_diabetes', 'Unknown')}
- Device: {user_info.get('device_type', 'Unknown')}

Current Glucose Data (14-day period):
- Average glucose: {stats.get('average_glucose', 0):.1f} mg/dL
- Time in range (70-180): {stats.get('time_in_range_70_180', 0):.1f}%
- Time below 70: {stats.get('time_below_70', 0):.1f}%
- Time above 180: {stats.get('time_above_180', 0):.1f}%
- Total readings: {stats.get('total_readings', 0)}
- Glucose variability (std): {stats.get('std_glucose', 0):.1f} mg/dL

Use this specific data when answering questions about the user's glucose patterns."""

        return base_prompt + context_addition
    
    def chat_with_mistral(self, user_message: str, prefer_agent: bool = False) -> Dict[str, Any]:
        """Main chat function with robust error handling"""
        if not user_message.strip():
            return {"success": False, "error": "Please enter a message"}
        
        try:
            # Get current context
            context = self.get_context_summary()
            
            # Build system prompt
            system_prompt = self.build_system_prompt(context)
            
            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (limited)
            if self.conversation_history:
                recent_history = self.conversation_history[-self.max_history:]
                messages.extend(recent_history)
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Try agent first if preferred and available
            if prefer_agent:
                agent_result = self.mistral_client.agent_completion(messages)
                if agent_result["success"]:
                    self._update_conversation_history(user_message, agent_result["response"])
                    return {
                        "success": True,
                        "response": agent_result["response"],
                        "method": "agent",
                        "context_included": not context.get("error")
                    }
                else:
                    self.logger.warning(f"Agent failed, trying chat completion: {agent_result['error']}")
            
            # Use chat completion API
            chat_result = self.mistral_client.chat_completion(messages)
            
            if chat_result["success"]:
                self._update_conversation_history(user_message, chat_result["response"])
                return {
                    "success": True,
                    "response": chat_result["response"],
                    "method": "chat_completion",
                    "context_included": not context.get("error"),
                    "usage": chat_result.get("usage", {})
                }
            else:
                return {
                    "success": False,
                    "error": chat_result["error"]
                }
            
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            return {
                "success": False,
                "error": f"Unexpected chat error: {str(e)}"
            }
    
    def _update_conversation_history(self, user_message: str, assistant_response: str):
        """Update conversation history"""
        self.conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Keep only recent messages
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        api_status = self.test_connection()
        config = get_config()
        
        return {
            "api_connected": api_status["success"],
            "api_message": api_status["message"],
            "user_loaded": self.current_user is not None,
            "data_available": self.current_glucose_data is not None and not self.current_glucose_data.empty,
            "conversation_messages": len(self.conversation_history),
            "current_user": self.current_user.name if self.current_user else None,
            "environment": config.app_environment,
            "hugging_face_space": config.is_huggingface_space,
            "agent_available": bool(config.mistral_agent_id)
        }

def create_secure_chat_instance() -> GlucoBuddyMistralChat:
    """Create a secure chat instance with proper configuration validation"""
    try:
        # Validate environment
        if not validate_environment():
            raise ValueError("Environment validation failed")
        
        # Create chat instance (will use environment variables)
        chat = GlucoBuddyMistralChat()
        
        # Test connection
        connection_test = chat.test_connection()
        if not connection_test["success"]:
            logger.warning(f"API connection test failed: {connection_test['message']}")
        
        return chat
        
    except Exception as e:
        logger.error(f"Failed to create secure chat instance: {e}")
        raise

def create_enhanced_cli():
    """Enhanced command-line interface with secure configuration"""
    print("🩺 GlucoBuddy Enhanced Chat Interface (Secure)")
    print("=" * 50)
    
    try:
        # Initialize chat system with security validation
        chat = create_secure_chat_instance()
        
        print("✅ Secure chat system initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize chat system: {e}")
        print("\n💡 Please check your environment configuration:")
        print("   - Ensure MISTRAL_API_KEY is set")
        print("   - Check that your API key is valid")
        return
    
    # Test connection
    print("\n🔍 Testing Mistral API connection...")
    connection_test = chat.test_connection()
    
    if connection_test["success"]:
        print(f"✅ {connection_test['message']}")
    else:
        print(f"❌ {connection_test['message']}")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return
    
    print("\n📋 Available commands:")
    print("  /load <user_key>  - Load demo user data")
    print("  /users           - List available demo users")
    print("  /status          - Show system status")
    print("  /clear           - Clear conversation history")
    print("  /test            - Test API connection")
    print("  /help            - Show this help")
    print("  /quit            - Exit")
    print("\n💬 Or just type your glucose-related questions!")
    print("\n" + "=" * 50)
    
    while True:
        try:
            user_input = input("\n🫵 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command_parts = user_input[1:].split()
                command = command_parts[0].lower()
                
                if command == 'quit':
                    print("\n👋 Thanks for using GlucoBuddy! Stay healthy! 🌟")
                    break
                
                elif command == 'help':
                    print("\n📋 Commands:")
                    print("  /load <user>  - Load user (sarah_g7, marcus_one, jennifer_g6, robert_receiver)")
                    print("  /users        - Show all available users")
                    print("  /status       - System status")
                    print("  /clear        - Clear chat history")
                    print("  /test         - Test API")
                    print("  /quit         - Exit")
                    continue
                
                elif command == 'clear':
                    chat.clear_conversation()
                    print("🧹 Conversation cleared!")
                    continue
                
                elif command == 'status':
                    status = chat.get_status()
                    print(f"\n📊 System Status:")
                    print(f"  🌐 API Connected: {'✅' if status['api_connected'] else '❌'} {status['api_message']}")
                    print(f"  👤 User Loaded: {'✅' if status['user_loaded'] else '❌'} {status.get('current_user', 'None')}")
                    print(f"  📊 Data Available: {'✅' if status['data_available'] else '❌'}")
                    print(f"  💬 Messages in Chat: {status['conversation_messages']}")
                    print(f"  🏠 Environment: {status['environment']}")
                    print(f"  🤗 Hugging Face Space: {'✅' if status['hugging_face_space'] else '❌'}")
                    print(f"  🤖 Agent Available: {'✅' if status['agent_available'] else '❌'}")
                    continue
                
                elif command == 'test':
                    print("🔍 Testing connection...")
                    test_result = chat.test_connection()
                    print(f"{'✅' if test_result['success'] else '❌'} {test_result['message']}")
                    continue
                
                elif command == 'users':
                    print("\n👥 Available Demo Users:")
                    for key, user in DEMO_USERS.items():
                        print(f"  📱 {key}: {user.name} ({user.diabetes_type}, {user.device_type})")
                    print("\n💡 Use: /load <user_key> to load their data")
                    continue
                
                elif command == 'load':
                    if len(command_parts) > 1:
                        user_key = command_parts[1]
                        print(f"📥 Loading data for {user_key}...")
                        
                        result = chat.load_user_data(user_key)
                        print(f"{result['message']}")
                        
                        if result['success']:
                            print(f"📊 {result['summary']}")
                    else:
                        print("❌ Usage: /load <user_key>")
                        print("💡 Use /users to see available users")
                    continue
                
                else:
                    print(f"❌ Unknown command: /{command}")
                    print("💡 Use /help to see available commands")
                    continue
            
            # Regular chat message
            print("🤔 Processing your question...")
            
            # Send to Mistral
            result = chat.chat_with_mistral(user_input, prefer_agent=True)
            
            if result['success']:
                method_info = f" [{result.get('method', 'unknown')}]"
                print(f"\n🤖 GlucoBuddy{method_info}: {result['response']}")
                
                # Show context info if no user data
                if not result.get('context_included', True):
                    print("\n💡 Tip: Load user data with '/load <user_key>' for personalized insights!")
                
                # Show usage info if available
                usage = result.get('usage', {})
                if usage:
                    tokens = usage.get('total_tokens', 0)
                    if tokens > 0:
                        print(f"\n📊 Tokens used: {tokens}")
            else:
                print(f"\n❌ Error: {result['error']}")
                
                # Provide helpful suggestions based on error type
                error_msg = result['error'].lower()
                if 'api key' in error_msg or '401' in error_msg:
                    print("💡 Check your Mistral API key configuration")
                elif 'rate limit' in error_msg or '429' in error_msg:
                    print("💡 Rate limit reached - please wait a moment before trying again")
                elif 'timeout' in error_msg:
                    print("💡 Request timed out - please try again")
                else:
                    print("💡 Use /test to check your connection")
        
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using GlucoBuddy! Take care! 🌟")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Try /status to check system state")

def run_quick_demo():
    """Run a quick demonstration of the secure system"""
    print("🚀 GlucoBuddy Quick Demo (Secure)")
    print("=" * 30)
    
    try:
        chat = create_secure_chat_instance()
        print("✅ Secure system initialized")
    except Exception as e:
        print(f"❌ Cannot run demo: {e}")
        return
    
    # Test connection
    print("\n1️⃣ Testing API connection...")
    connection = chat.test_connection()
    print(f"   {'✅' if connection['success'] else '❌'} {connection['message']}")
    
    if not connection['success']:
        print("❌ Cannot proceed without API connection")
        return
    
    # Load demo user
    print("\n2️⃣ Loading demo user (Sarah)...")
    load_result = chat.load_user_data("sarah_g7")
    print(f"   {load_result['message']}")
    
    if load_result['success']:
        print(f"   📊 {load_result['summary']}")
    
    # Demo chat
    if load_result['success']:
        print("\n3️⃣ Demo conversation...")
        
        demo_questions = [
            "What's my average glucose level?",
            "How is my time in range?",
            "Any recommendations for improvement?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n   Q{i}: {question}")
            result = chat.chat_with_mistral(question)
            
            if result['success']:
                response = result['response']
                # Truncate long responses for demo
                if len(response) > 200:
                    response = response[:200] + "..."
                print(f"   A{i}: {response}")
            else:
                print(f"   A{i}: ❌ {result['error']}")
    
    print(f"\n✅ Demo completed! Use the full CLI with create_enhanced_cli()")

def main():
    """Main function with secure configuration menu system"""
    print("🩺 GlucoBuddy - AI-Powered Glucose Chat Assistant (Secure)")
    print("=" * 60)
    
    # Validate configuration first
    print("🔍 Validating configuration...")
    if not validate_environment():
        print("\n❌ Configuration validation failed!")
        print("Please set up your environment variables before continuing.")
        return
    
    print("✅ Configuration validation passed!")
    
    print("\n🎯 Choose an option:")
    print("1. 💬 Start interactive chat")
    print("2. 🚀 Run quick demo")
    print("3. 🔧 Show configuration")
    print("4. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                create_enhanced_cli()
                break
            elif choice == '2':
                run_quick_demo()
                break
            elif choice == '3':
                validate_environment()
                break
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()