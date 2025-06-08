#!/usr/bin/env python3
"""
GlucoBuddy Mistral Chat Integration
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

# Import your existing classes
from apifunctions import DexcomAPI, GlucoseAnalyzer, DEMO_USERS, DemoUser

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
                from apifunctions import GlucoseMockData
                egv_data = GlucoseMockData.generate_realistic_glucose_data(7)
            
            # Process data
            self.current_glucose_data = self.analyzer.process_egv_data(egv_data)
            self.current_stats = self.analyzer.calculate_basic_stats(self.current_glucose_data)
            self.current_patterns = self.analyzer.identify_patterns(self.current_glucose_data)
            
            return {
                "success": True,
                "message": f"✅ Successfully loaded data for {self.current_user.name}",
                "user": asdict(self.current_user),
                "data_points": len(self.current_glucose_data),
                "stats": self.current_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load user data: {e}")
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
            "current_stats": self.current_stats,
            "patterns": self.current_patterns,
            "data_points": len(self.current_glucose_data) if self.current_glucose_data is not None else 0
        }
        
        # Add recent readings
        if self.current_glucose_data is not None and not self.current_glucose_data.empty:
            recent_data = self.current_glucose_data.tail(5)
            context["recent_readings"] = []
            for _, row in recent_data.iterrows():
                context["recent_readings"].append({
                    "time": pd.to_datetime(row['displayTime']).isoformat(),
                    "glucose": row['value'],
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
            """
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Prepare request payload
            payload = {
                "agent_id": self.mistral_agent_id,
                "messages": [
                    {"role": "system", "content": system_prompt}
                ] + self.conversation_history[-10:],  # Keep last 10 messages
                "max_tokens": 500,
                "temperature": 0.7
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
                    "error": f"API request failed: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error calling Mistral API: {e}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}"
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

def create_chat_interface():
    """Simple command-line chat interface"""
    print("🩺 GlucoBuddy Chat Interface")
    print("=" * 50)
    
    # Get Mistral API configuration
    api_key = input("Enter your Mistral API key: ").strip()
    agent_id = input("Enter your Mistral Agent ID (optional): ").strip() or None
    
    if not api_key:
        print("❌ API key is required!")
        return
    
    chat = GlucoBuddyMistralChat(api_key, agent_id)
    
    print("\n📋 Available commands:")
    print("  /load <user_key> - Load demo user data")
    print("  /users - List available users") 
    print("  /clear - Clear conversation")
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
            result = chat.chat_with_mistral(user_input)
            
            if result['success']:
                print(f"\nGlucoBuddy: {result['response']}\n")
                if not result.get('context_included'):
                    print("💡 Tip: Load user data with /load <user_key> for personalized advice!\n")
            else:
                print(f"❌ Error: {result['error']}\n")
                
        except KeyboardInterrupt:
            print("\n👋 Thanks for using GlucoBuddy!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

# Alternative: Flask web interface
def create_web_interface():
    """Create a simple web interface using Flask"""
    try:
        from flask import Flask, request, jsonify, render_template_string
        
        app = Flask(__name__)
        chat_instance = None
        
        @app.route('/')
        def index():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GlucoBuddy Chat</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
                    .message { margin: 5px 0; padding: 8px; border-radius: 5px; }
                    .user { background-color: #e3f2fd; text-align: right; }
                    .assistant { background-color: #f1f8e9; }
                    .controls { margin: 10px 0; }
                    input[type="text"] { width: 70%; padding: 8px; }
                    button { padding: 8px 15px; margin: 0 5px; }
                </style>
            </head>
            <body>
                <h1>🩺 GlucoBuddy Chat</h1>
                <div class="controls">
                    <input type="text" id="apiKey" placeholder="Enter Mistral API Key" />
                    <button onclick="initializeChat()">Initialize</button>
                </div>
                <div class="controls">
                    <select id="userSelect">
                        <option value="">Select Demo User</option>
                        <option value="sarah_g7">Sarah (G7)</option>
                        <option value="marcus_one">Marcus (G7)</option>
                        <option value="jennifer_g6">Jennifer (G6)</option>
                        <option value="robert_receiver">Robert (Receiver)</option>
                    </select>
                    <button onclick="loadUser()">Load User Data</button>
                </div>
                <div id="chatContainer" class="chat-container"></div>
                <div class="controls">
                    <input type="text" id="messageInput" placeholder="Ask about glucose management..." />
                    <button onclick="sendMessage()">Send</button>
                    <button onclick="clearChat()">Clear</button>
                </div>
                
                <script>
                    let chatInitialized = false;
                    
                    function initializeChat() {
                        const apiKey = document.getElementById('apiKey').value;
                        if (!apiKey) {
                            alert('Please enter your Mistral API key');
                            return;
                        }
                        
                        fetch('/initialize', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({api_key: apiKey})
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                chatInitialized = true;
                                addMessage('System', '✅ Chat initialized! You can now load user data and ask questions.');
                            } else {
                                alert('Failed to initialize: ' + data.error);
                            }
                        });
                    }
                    
                    function loadUser() {
                        if (!chatInitialized) {
                            alert('Please initialize chat first');
                            return;
                        }
                        
                        const userKey = document.getElementById('userSelect').value;
                        if (!userKey) {
                            alert('Please select a user');
                            return;
                        }
                        
                        fetch('/load_user', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({user_key: userKey})
                        })
                        .then(response => response.json())
                        .then(data => {
                            addMessage('System', data.message);
                        });
                    }
                    
                    function sendMessage() {
                        if (!chatInitialized) {
                            alert('Please initialize chat first');
                            return;
                        }
                        
                        const messageInput = document.getElementById('messageInput');
                        const message = messageInput.value.trim();
                        if (!message) return;
                        
                        addMessage('You', message);
                        messageInput.value = '';
                        
                        fetch('/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: message})
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                addMessage('GlucoBuddy', data.response);
                            } else {
                                addMessage('Error', data.error);
                            }
                        });
                    }
                    
                    function clearChat() {
                        if (chatInitialized) {
                            fetch('/clear', {method: 'POST'});
                        }
                        document.getElementById('chatContainer').innerHTML = '';
                    }
                    
                    function addMessage(sender, text) {
                        const container = document.getElementById('chatContainer');
                        const messageDiv = document.createElement('div');
                        messageDiv.className = 'message ' + (sender === 'You' ? 'user' : 'assistant');
                        messageDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
                        container.appendChild(messageDiv);
                        container.scrollTop = container.scrollHeight;
                    }
                    
                    // Enter key support
                    document.getElementById('messageInput').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            sendMessage();
                        }
                    });
                </script>
            </body>
            </html>
            """)
        
        @app.route('/initialize', methods=['POST'])
        def initialize():
            global chat_instance
            data = request.json
            api_key = data.get('api_key')
            
            if not api_key:
                return jsonify({"success": False, "error": "API key required"})
            
            try:
                chat_instance = GlucoBuddyMistralChat(api_key)
                return jsonify({"success": True})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        @app.route('/load_user', methods=['POST'])
        def load_user():
            if chat_instance is None:
                return jsonify({"success": False, "message": "Chat not initialized"})
            
            data = request.json
            user_key = data.get('user_key')
            result = chat_instance.load_user_data(user_key)
            return jsonify(result)
        
        @app.route('/chat', methods=['POST'])
        def chat():
            if chat_instance is None:
                return jsonify({"success": False, "error": "Chat not initialized"})
            
            data = request.json
            message = data.get('message')
            result = chat_instance.chat_with_mistral(message)
            return jsonify(result)
        
        @app.route('/clear', methods=['POST'])
        def clear():
            if chat_instance:
                chat_instance.clear_conversation()
            return jsonify({"success": True})
        
        print("🌐 Starting web interface at http://localhost:5000")
        app.run(debug=True, port=5000)
        
    except ImportError:
        print("❌ Flask not installed. Install with: pip install flask")
        print("💡 Using command-line interface instead...")
        create_chat_interface()

if __name__ == "__main__":
    # Choose interface
    interface_choice = input("Choose interface (1: Command-line, 2: Web): ").strip()
    
    if interface_choice == "2":
        create_web_interface()
    else:
        create_chat_interface()