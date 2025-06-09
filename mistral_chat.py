#!/usr/bin/env python3
"""
GlucoBuddy Mistral Chat Integration - Fixed JSON Serialization
Simple chat functionality using Mistral agents for glucose data analysis
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

from apifunctions import (
    DexcomAPI,
    GlucoseAnalyzer,
    DEMO_USERS,
    DemoUser,
    format_glucose_data_for_display
)

# Initialize Mistral chat with your agent ID
MISTRAL_API_KEY = "ZAjtPftvZrCxK7WWwjBJIYudaiNhwRuO"
MISTRAL_AGENT_ID = "ag:2d7a33b1:20250608:glycoaiagent:cc72ded9"

class GlucoseMockData:
    """Generate realistic mock glucose data for testing"""
    
    @staticmethod
    def generate_realistic_glucose_data(days: int = 7) -> List[Dict]:
        """Generate realistic glucose data patterns"""
        data = []
        start_time = datetime.now() - timedelta(days=days)
        
        # Generate data points every 5 minutes
        current_time = start_time
        base_glucose = 120  # Starting glucose level
        
        while current_time <= datetime.now():
            # Simulate realistic glucose patterns
            hour = current_time.hour
            
            # Dawn phenomenon (higher glucose in early morning)
            if 4 <= hour <= 8:
                base_adjustment = random.uniform(10, 30)
            # Post-meal spikes
            elif hour in [8, 12, 18]:  # Breakfast, lunch, dinner times
                base_adjustment = random.uniform(20, 60)
            # Normal variation
            else:
                base_adjustment = random.uniform(-20, 20)
            
            # Add some random noise
            noise = random.uniform(-15, 15)
            glucose_value = max(60, min(300, base_glucose + base_adjustment + noise))
            
            # Determine trend based on recent changes
            trend_options = ['flat', 'fortyFiveUp', 'singleUp', 'fortyFiveDown', 'singleDown']
            trend = random.choice(trend_options)
            
            data_point = {
                'systemTime': current_time.isoformat(),
                'displayTime': current_time.isoformat(),
                'value': round(glucose_value),
                'trend': trend,
                'realtimeValue': round(glucose_value),
                'smoothedValue': round(glucose_value)
            }
            
            data.append(data_point)
            
            # Move to next data point (5 minutes later)
            current_time += timedelta(minutes=5)
            
            # Gradually adjust base glucose for next reading
            base_glucose = glucose_value * 0.9 + 120 * 0.1  # Mean reversion
        
        return data
    
    @staticmethod
    def generate_user_specific_data(user: DemoUser, days: int = 7) -> List[Dict]:
        """Generate glucose data specific to user patterns"""
        base_data = GlucoseMockData.generate_realistic_glucose_data(days)
        
        # Modify data based on user characteristics
        if user.typical_glucose_pattern == "stable_with_meal_spikes":
            # Reduce overall variability but keep meal spikes
            for point in base_data:
                hour = datetime.fromisoformat(point['systemTime']).hour
                if hour not in [8, 12, 18]:  # Non-meal times
                    point['value'] = int(point['value'] * 0.8 + 110 * 0.2)
        
        elif user.typical_glucose_pattern == "exercise_related_lows":
            # Add some low glucose episodes
            for i, point in enumerate(base_data):
                if i % 50 == 0:  # Occasional lows
                    point['value'] = max(60, int(point['value'] * 0.6))
                    point['trend'] = 'singleDown'
        
        elif user.typical_glucose_pattern == "dawn_phenomenon":
            # Emphasize morning highs
            for point in base_data:
                hour = datetime.fromisoformat(point['systemTime']).hour
                if 4 <= hour <= 8:
                    point['value'] = min(250, int(point['value'] * 1.3))
        
        return base_data

def convert_numpy_types(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, pd.Int64Dtype)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class GlucoBuddyMistralChat:
    """Simple chat interface using Mistral agents for glucose monitoring"""
    
    def __init__(self, mistral_api_key: str, mistral_agent_id: str = None):
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        self.current_user: Optional[DemoUser] = None
        self.current_glucose_data: Optional[pd.DataFrame] = None
        self.current_stats: Optional[Dict] = None
        self.current_patterns: Optional[Dict] = None
        
        # Mistral configuration
        self.mistral_api_key = mistral_api_key
        self.mistral_agent_id = mistral_agent_id or "your-glucose-agent-id"
        self.mistral_api_url = "https://api.mistral.ai/v1/agents/completions"
        
        # Setup logging  
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Chat conversation history
        self.conversation_history = []

    def load_user_data(self, user_key: str) -> Dict[str, Any]:
        """Load glucose data for a specific demo user"""
        try:
            if user_key not in DEMO_USERS:
                return {
                    "success": False,
                    "message": f"❌ Invalid user key. Available: {', '.join(DEMO_USERS.keys())}"
                }
            
            self.current_user = DEMO_USERS[user_key]
            
            # Authenticate and load data
            self.dexcom_api.simulate_demo_login(user_key)
            
            # Get glucose data (this will use mock data if API fails)
            egv_data = self.dexcom_api.get_egv_data()
            
            # If no real data, generate mock data
            if not egv_data:
                self.logger.info("No API data available, generating mock data")
                egv_data = GlucoseMockData.generate_user_specific_data(self.current_user, 7)
            
            # Process data
            self.current_glucose_data = self.analyzer.process_egv_data(egv_data)
            self.current_stats = self.analyzer.calculate_basic_stats(self.current_glucose_data)
            self.current_patterns = self.analyzer.identify_patterns(self.current_glucose_data)
            
            return {
                "success": True,
                "message": f"✅ Successfully loaded data for {self.current_user.name}",
                "user": asdict(self.current_user),
                "data_points": len(self.current_glucose_data),
                "stats": convert_numpy_types(self.current_stats)  # Convert here too
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load user data: {e}")
            # Try to generate mock data as fallback
            try:
                if user_key in DEMO_USERS:
                    self.current_user = DEMO_USERS[user_key]
                    egv_data = GlucoseMockData.generate_user_specific_data(self.current_user, 7)
                    self.current_glucose_data = self.analyzer.process_egv_data(egv_data)
                    self.current_stats = self.analyzer.calculate_basic_stats(self.current_glucose_data)
                    self.current_patterns = self.analyzer.identify_patterns(self.current_glucose_data)
                    
                    return {
                        "success": True,
                        "message": f"✅ Successfully loaded mock data for {self.current_user.name}",
                        "user": asdict(self.current_user),
                        "data_points": len(self.current_glucose_data),
                        "stats": convert_numpy_types(self.current_stats)
                    }
            except Exception as fallback_error:
                self.logger.error(f"Fallback mock data generation failed: {fallback_error}")
            
            return {
                "success": False,
                "message": f"❌ Failed to load user data: {str(e)}"
            }

    def get_current_context(self) -> Dict[str, Any]:
        """Get current glucose data context for the chat agent"""
        if not self.current_user or not self.current_stats:
            return {"error": "No user data loaded"}
        
        context = {
            "user": {
                "name": self.current_user.name,
                "age": self.current_user.age,
                "diabetes_type": self.current_user.diabetes_type,
                "device_type": self.current_user.device_type,
                "years_with_diabetes": self.current_user.years_with_diabetes
            },
            "current_stats": convert_numpy_types(self.current_stats),  # Convert numpy types
            "patterns": convert_numpy_types(self.current_patterns),    # Convert numpy types
            "data_points": len(self.current_glucose_data) if self.current_glucose_data is not None else 0
        }
        
        # Add recent readings
        if self.current_glucose_data is not None and not self.current_glucose_data.empty:
            recent_data = self.current_glucose_data.tail(5)
            context["recent_readings"] = []
            for _, row in recent_data.iterrows():
                context["recent_readings"].append({
                    "time": pd.to_datetime(row['displayTime']).isoformat(),
                    "glucose": convert_numpy_types(row['value']),  # Convert numpy type
                    "trend": row.get('trend', 'flat')
                })
        
        return context

    def chat_with_mistral(self, user_message: str) -> Dict[str, Any]:
        """Send message to Mistral agent and get response"""
        try:
            # Get current glucose context
            context = self.get_current_context()
            
            # Prepare the message with context
            system_prompt = f"""You are GlucoBuddy, a helpful diabetes management assistant. 
            You have access to the user's glucose data and should provide personalized advice.
            
            Current user context: {json.dumps(context, indent=2)}
            
            Guidelines:
            - Be supportive and encouraging
            - Provide actionable glucose management advice
            - Reference specific data when relevant
            - Always recommend consulting healthcare providers for medical decisions
            - Use emojis to make responses friendly
            - Try to keep the answer conversational, when the user asks for more details abput your answer. Just provide that detail.
            - Provide answers that take into account your max tokens
            - Don't make up data: always refer to data loaded from Dexcom
            - When data is available from Dexcom, don't calculate it yourself (example: Time in range)
            """
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Prepare request payload
            payload = {
                "agent_id": self.mistral_agent_id,
                "messages": [
                    {"role": "system", "content": system_prompt}
                ] + self.conversation_history[-10:],  # Keep last 10 messages
                "max_tokens": 1000
            }
            
            # Make API request to Mistral
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.mistral_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return {
                    "success": True,
                    "response": assistant_message,
                    "context_included": context.get("error") is None
                }
            else:
                self.logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API request failed: {response.status_code} - {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error calling Mistral API: {e}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}"
            }
        except KeyError as e:
            self.logger.error(f"Unexpected API response format: {e}")
            return {
                "success": False,
                "error": f"Unexpected API response format - missing key: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in chat: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "messages": len(self.conversation_history),
            "user_loaded": self.current_user is not None,
            "data_available": self.current_glucose_data is not None
        }

    def chat_with_direct_model(self, user_message: str, model: str = "mistral-large-latest") -> Dict[str, Any]:
        """Alternative method: Use direct model API instead of agent (allows temperature control)"""
        try:
            context = self.get_current_context()
            
            system_prompt = f"""You are GlucoBuddy, a helpful diabetes management assistant. 
            You have access to the user's glucose data and should provide personalized advice.
            
            Current user context: {json.dumps(context, indent=2)}
            
            Guidelines:
            - Be supportive and encouraging
            - Provide actionable glucose management advice
            - Reference specific data when relevant
            - Always recommend consulting healthcare providers for medical decisions
            - Use emojis to make responses friendly
            """
            
            # For direct model calls, use different endpoint
            direct_api_url = "https://api.mistral.ai/v1/chat/completions"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ] + self.conversation_history[-8:],  # Keep last 8 messages for context
                "max_tokens": 500,
                "temperature": 0.7  # This works with direct model calls
            }
            
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                direct_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return {
                    "success": True,
                    "response": assistant_message,
                    "context_included": context.get("error") is None,
                    "method": "direct_model"
                }
            else:
                return {
                    "success": False,
                    "error": f"Direct model API request failed: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Direct model error: {str(e)}"
            }

def create_chat_interface():
    """Simple command-line chat interface"""
    print("🩺 GlucoBuddy Chat Interface")
    print("=" * 50)
    
    # Get Mistral API configuration
    api_key = input("Enter your Mistral API key (or press Enter to use default): ").strip()
    if not api_key:
        api_key = MISTRAL_API_KEY
        
    agent_id = input("Enter your Mistral Agent ID (or press Enter to use default): ").strip()
    if not agent_id:
        agent_id = MISTRAL_AGENT_ID
        
    # Ask for chat method
    print("\nChoose chat method:")
    print("1. Use Mistral Agent (no temperature control)")
    print("2. Use Direct Model API (with temperature control)")
    method_choice = input("Enter choice (1 or 2): ").strip()
    use_agent = method_choice != "2"
    
    if not api_key:
        print("❌ API key is required!")
        return
    
    chat = GlucoBuddyMistralChat(api_key, agent_id)
    
    print(f"\n🤖 Using {'Agent' if use_agent else 'Direct Model'} method")
    print("\n📋 Available commands:")
    print("  /load <user_key> - Load demo user data")
    print("  /users - List available users") 
    print("  /clear - Clear conversation")
    print("  /switch - Switch between agent and direct model")
    print("  /quit - Exit chat")
    print("  Or just type your glucose-related question!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith('/'):
                command_parts = user_input[1:].split()
                command = command_parts[0].lower()
                
                if command == 'quit':
                    print("👋 Thanks for using GlucoBuddy!")
                    break
                    
                elif command == 'clear':
                    chat.clear_conversation()
                    print("🧹 Conversation cleared!")
                    continue
                    
                elif command == 'switch':
                    use_agent = not use_agent
                    method = "Agent" if use_agent else "Direct Model"
                    print(f"🔄 Switched to {method} method")
                    continue
                    
                elif command == 'users':
                    print("\n👥 Available Demo Users:")
                    for key, user in DEMO_USERS.items():
                        print(f"  {key}: {user.name} ({user.diabetes_type})")
                    print()
                    continue
                    
                elif command == 'load':
                    if len(command_parts) > 1:
                        user_key = command_parts[1]
                        result = chat.load_user_data(user_key)
                        print(f"\n{result['message']}")
                        if result['success']:
                            stats = result['stats']
                            print(f"📊 Average glucose: {stats.get('average_glucose', 0):.1f} mg/dL")
                            print(f"🎯 Time in range: {stats.get('time_in_range_70_180', 0):.1f}%\n")
                    else:
                        print("❌ Usage: /load <user_key>")
                    continue
                    
                else:
                    print(f"❌ Unknown command: {command}")
                    continue
            
            # Regular chat message
            print("🤔 Thinking...")
            
            if use_agent:
                result = chat.chat_with_mistral(user_input)
            else:
                result = chat.chat_with_direct_model(user_input)
            
            if result['success']:
                method_info = f" [{result.get('method', 'agent')}]" if not use_agent else ""
                print(f"\nGlucoBuddy{method_info}: {result['response']}\n")
                if not result.get('context_included'):
                    print("💡 Tip: Load user data with /load <user_key> for personalized advice!\n")
            else:
                print(f"❌ Error: {result['error']}\n")
                if "422" in str(result['error']) and use_agent:
                    print("💡 Try switching to direct model with /switch command\n")
                
        except KeyboardInterrupt:
            print("\n👋 Thanks for using GlucoBuddy!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    create_chat_interface()