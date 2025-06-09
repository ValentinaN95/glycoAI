#!/usr/bin/env python3
"""
Hybrid Dexcom Integration
Combines demo users with optional real Dexcom authentication
"""

import os
import gradio as gr
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import random

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DemoUser:
    """Enhanced demo user with auth type"""
    name: str
    device_type: str
    username: str
    password: str
    description: str
    age: int = 30
    diabetes_type: str = "Type 1"
    years_with_diabetes: int = 5
    typical_glucose_pattern: str = "normal"
    auth_type: str = "demo"  # "demo" or "real"
    
# Enhanced demo users + real auth option
ENHANCED_DEMO_USERS = {
    # Your existing 4 demo users (unchanged for easy demos)
    "sarah_demo": DemoUser(
        name="Sarah Thompson (Demo)",
        age=32,
        device_type="G7 Mobile App",
        username="demo_sarah",
        password="demo123",
        description="Demo: Active professional with Type 1 diabetes, stable glucose control",
        diabetes_type="Type 1",
        years_with_diabetes=8,
        typical_glucose_pattern="stable_with_meal_spikes",
        auth_type="demo"
    ),
    "marcus_demo": DemoUser(
        name="Marcus Rodriguez (Demo)",
        age=45,
        device_type="ONE+ Mobile App", 
        username="demo_marcus",
        password="demo123",
        description="Demo: Father with Type 2 diabetes, moderate variability",
        diabetes_type="Type 2",
        years_with_diabetes=3,
        typical_glucose_pattern="moderate_variability",
        auth_type="demo"
    ),
    "jennifer_demo": DemoUser(
        name="Jennifer Chen (Demo)",
        age=28,
        device_type="G6 Mobile App",
        username="demo_jennifer",
        password="demo123",
        description="Demo: Graduate student with Type 1, athletic lifestyle",
        diabetes_type="Type 1",
        years_with_diabetes=12,
        typical_glucose_pattern="exercise_related_lows",
        auth_type="demo"
    ),
    "robert_demo": DemoUser(
        name="Robert Williams (Demo)",
        age=67,
        device_type="G6 Touchscreen Receiver",
        username="demo_robert",
        password="demo123",
        description="Demo: Retired teacher with Type 2, prefers receiver device",
        diabetes_type="Type 2",
        years_with_diabetes=15,
        typical_glucose_pattern="dawn_phenomenon",
        auth_type="demo"
    ),
    
    # NEW: Real authentication option
    "real_user": DemoUser(
        name="Real Dexcom User",
        age=0,  # Will be determined from real data
        device_type="Real Dexcom Device",
        username="real_dexcom_auth",
        password="oauth_flow",
        description="Authenticate with your real Dexcom account using OAuth",
        diabetes_type="Real Data",
        years_with_diabetes=0,
        typical_glucose_pattern="real_data",
        auth_type="real"
    )
}

class HybridDexcomManager:
    """Manages both demo and real Dexcom authentication"""
    
    def __init__(self):
        self.demo_enabled = True
        self.real_auth_enabled = self._check_real_auth_available()
        self.current_mode = "demo"
        
        # Initialize real auth if available
        if self.real_auth_enabled:
            try:
                from dexcom_real_auth_system import DexcomRealAPI
                self.real_api = DexcomRealAPI(environment="sandbox")
                logger.info("✅ Real Dexcom authentication available")
            except ImportError:
                logger.warning("⚠️ Real Dexcom auth not available - missing module")
                self.real_auth_enabled = False
            except Exception as e:
                logger.warning(f"⚠️ Real Dexcom auth not available: {e}")
                self.real_auth_enabled = False
        
        # Mock data generator for demo users
        self.mock_generator = MockGlucoseGenerator()
    
    def _check_real_auth_available(self) -> bool:
        """Check if real authentication is properly configured"""
        client_id = os.getenv("DEXCOM_CLIENT_ID")
        client_secret = os.getenv("DEXCOM_CLIENT_SECRET")
        
        # Also check for hardcoded values in the real auth system
        try:
            from dexcom_real_auth_system import CLIENT_ID, CLIENT_SECRET
            if CLIENT_ID and CLIENT_ID != "YOUR_REAL_CLIENT_ID_HERE":
                return True
        except ImportError:
            pass
        
        return bool(client_id and client_secret)
    
    def get_user_options(self) -> Dict[str, str]:
        """Get available user options for the UI"""
        options = {}
        
        # Add demo users
        for key, user in ENHANCED_DEMO_USERS.items():
            if user.auth_type == "demo":
                options[key] = f"🎭 {user.name}"
        
        # Add real auth option if available
        if self.real_auth_enabled:
            options["real_user"] = "🔐 Real Dexcom User (OAuth)"
        else:
            options["real_user_disabled"] = "🔒 Real Dexcom User (Configure to Enable)"
        
        return options
    
    def authenticate_user(self, user_key: str) -> Dict[str, Any]:
        """Authenticate user (demo or real)"""
        if user_key not in ENHANCED_DEMO_USERS:
            return {"success": False, "message": "Invalid user selection"}
        
        user = ENHANCED_DEMO_USERS[user_key]
        
        if user.auth_type == "demo":
            return self._authenticate_demo_user(user_key, user)
        elif user.auth_type == "real":
            return self._authenticate_real_user()
        else:
            return {"success": False, "message": "Unknown authentication type"}
    
    def _authenticate_demo_user(self, user_key: str, user: DemoUser) -> Dict[str, Any]:
        """Authenticate demo user (instant)"""
        try:
            # Generate mock data for demo user
            mock_data = self.mock_generator.generate_user_data(user)
            
            return {
                "success": True,
                "message": f"✅ Demo user authenticated: {user.name}",
                "user": user,
                "data": mock_data,
                "auth_type": "demo"
            }
        except Exception as e:
            return {"success": False, "message": f"Demo authentication failed: {e}"}
    
    def _authenticate_real_user(self) -> Dict[str, Any]:
        """Authenticate real Dexcom user"""
        if not self.real_auth_enabled:
            return {
                "success": False, 
                "message": "Real authentication not configured. Check DEXCOM_CLIENT_ID/SECRET"
            }
        
        try:
            # Start OAuth flow
            auth_success = self.real_api.start_oauth_flow()
            
            if auth_success:
                # Get real data
                real_data = self._fetch_real_data()
                
                return {
                    "success": True,
                    "message": "✅ Real Dexcom user authenticated",
                    "user": self._create_real_user_profile(),
                    "data": real_data,
                    "auth_type": "real"
                }
            else:
                return {"success": False, "message": "OAuth authentication failed"}
                
        except Exception as e:
            logger.error(f"Real authentication error: {e}")
            return {"success": False, "message": f"Real authentication failed: {e}"}
    
    def _fetch_real_data(self) -> Dict[str, Any]:
        """Fetch real data from Dexcom API"""
        try:
            # Get data range
            data_range = self.real_api.get_data_range()
            
            # Get glucose data (last 14 days)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=14)
            
            egv_data = self.real_api.get_egv_data(
                start_date=start_time.isoformat(),
                end_date=end_time.isoformat()
            )
            
            # Get events data
            events_data = self.real_api.get_events_data(
                start_date=start_time.isoformat(),
                end_date=end_time.isoformat()
            )
            
            return {
                "data_range": data_range,
                "egv_data": egv_data,
                "events_data": events_data,
                "source": "real_dexcom_api"
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch real data: {e}")
            return {"error": f"Failed to fetch real data: {e}"}
    
    def _create_real_user_profile(self) -> DemoUser:
        """Create user profile from real data"""
        return DemoUser(
            name="Real Dexcom User",
            age=0,  # Could extract from real user data if available
            device_type="Real Dexcom Device",
            username="authenticated_real_user",
            password="oauth_token",
            description="Authenticated real Dexcom user",
            diabetes_type="From Real Data",
            years_with_diabetes=0,
            typical_glucose_pattern="real_data",
            auth_type="real"
        )

class MockGlucoseGenerator:
    """Enhanced mock glucose data generator"""
    
    def generate_user_data(self, user: DemoUser, days: int = 14) -> Dict[str, Any]:
        """Generate mock data based on user profile"""
        # Generate glucose readings
        egv_data = self._generate_glucose_readings(user, days)
        
        # Generate events (meals, insulin)
        events_data = self._generate_events_data(user, days)
        
        # Create data range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        data_range = {
            "egvStart": start_time.isoformat(),
            "egvEnd": end_time.isoformat(),
            "eventStart": start_time.isoformat(),
            "eventEnd": end_time.isoformat()
        }
        
        return {
            "data_range": data_range,
            "egv_data": egv_data,
            "events_data": events_data,
            "source": "mock_data"
        }
    
    def _generate_glucose_readings(self, user: DemoUser, days: int) -> List[Dict]:
        """Generate realistic glucose readings"""
        import random
        import numpy as np
        
        readings = []
        start_time = datetime.now() - timedelta(days=days)
        
        # Base glucose level based on user pattern
        base_glucose = {
            "stable_with_meal_spikes": 120,
            "moderate_variability": 140,
            "exercise_related_lows": 115,
            "dawn_phenomenon": 130
        }.get(user.typical_glucose_pattern, 125)
        
        current_glucose = base_glucose
        
        # Generate readings every 5 minutes
        for i in range(days * 288):  # 288 readings per day
            timestamp = start_time + timedelta(minutes=i * 5)
            hour = timestamp.hour
            
            # Apply user-specific patterns
            target_glucose = self._calculate_target_glucose(user, hour, base_glucose)
            
            # Smooth glucose changes
            change = (target_glucose - current_glucose) * 0.2
            current_glucose += change + random.uniform(-5, 5)
            current_glucose = max(60, min(300, current_glucose))
            
            # Determine trend
            trend = self._calculate_trend(change)
            
            readings.append({
                "systemTime": timestamp.isoformat() + "Z",
                "displayTime": timestamp.isoformat() + "Z",
                "value": round(current_glucose),
                "trend": trend,
                "realtimeValue": round(current_glucose),
                "smoothedValue": round(current_glucose)
            })
        
        return readings
    
    def _calculate_target_glucose(self, user: DemoUser, hour: int, base: float) -> float:
        """Calculate target glucose based on user pattern and time"""
        if user.typical_glucose_pattern == "dawn_phenomenon":
            if 4 <= hour <= 8:
                return base + 40  # Dawn phenomenon spike
        elif user.typical_glucose_pattern == "exercise_related_lows":
            if 17 <= hour <= 19:  # Evening exercise
                return base - 30  # Exercise-induced low
        elif user.typical_glucose_pattern == "moderate_variability":
            return base + random.uniform(-20, 30)  # High variability
        
        # Standard meal patterns
        if 7 <= hour <= 9:  # Breakfast
            return base + random.uniform(20, 50)
        elif 12 <= hour <= 14:  # Lunch
            return base + random.uniform(25, 60)
        elif 18 <= hour <= 20:  # Dinner
            return base + random.uniform(30, 70)
        else:
            return base + random.uniform(-10, 15)
    
    def _calculate_trend(self, change: float) -> str:
        """Calculate trend arrow"""
        if change > 3:
            return "singleUp"
        elif change > 1:
            return "fortyFiveUp"
        elif change < -3:
            return "singleDown"
        elif change < -1:
            return "fortyFiveDown"
        else:
            return "flat"
    
    def _generate_events_data(self, user: DemoUser, days: int) -> List[Dict]:
        """Generate mock events (meals, insulin)"""
        import random
        
        events = []
        start_date = (datetime.now() - timedelta(days=days)).date()
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Generate daily meals and insulin
            for meal_time, meal_name in [(7, "breakfast"), (12, "lunch"), (18, "dinner")]:
                # Meal event
                meal_dt = datetime.combine(current_date, datetime.min.time().replace(
                    hour=meal_time, minute=random.randint(0, 30)
                ))
                
                carbs = random.randint(30, 80)
                events.append({
                    "systemTime": meal_dt.isoformat() + "Z",
                    "displayTime": meal_dt.isoformat() + "Z",
                    "eventType": "carbs",
                    "eventSubType": meal_name,
                    "value": carbs,
                    "unit": "grams"
                })
                
                # Insulin event (if Type 1)
                if user.diabetes_type == "Type 1":
                    insulin_dt = meal_dt + timedelta(minutes=random.randint(5, 15))
                    insulin_units = round(carbs / random.uniform(10, 15), 1)
                    
                    events.append({
                        "systemTime": insulin_dt.isoformat() + "Z",
                        "displayTime": insulin_dt.isoformat() + "Z",
                        "eventType": "insulin",
                        "eventSubType": "fast",
                        "value": insulin_units,
                        "unit": "units"
                    })
        
        return events

def create_hybrid_ui_components():
    """Create UI components for hybrid demo"""
    
    # Initialize the hybrid manager
    hybrid_manager = HybridDexcomManager()
    user_options = hybrid_manager.get_user_options()
    
    # Create user selection buttons
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👥 Select User Type")
            gr.Markdown("Choose from demo users (instant) or authenticate with real Dexcom account")
            
            # Demo users row
            with gr.Row():
                demo_buttons = []
                for key, user in ENHANCED_DEMO_USERS.items():
                    if user.auth_type == "demo":
                        btn = gr.Button(
                            f"🎭 {user.name.split('(')[0].strip()}\n{user.device_type}",
                            variant="secondary",
                            size="lg"
                        )
                        demo_buttons.append((key, btn))
            
            # Real auth button
            with gr.Row():
                if hybrid_manager.real_auth_enabled:
                    real_auth_btn = gr.Button(
                        "🔐 REAL DEXCOM USER\n(OAuth Authentication)",
                        variant="primary",
                        size="lg"
                    )
                else:
                    real_auth_btn = gr.Button(
                        "🔒 Real Dexcom (Not Configured)\nSet DEXCOM_CLIENT_ID/SECRET",
                        variant="secondary",
                        size="lg",
                        interactive=False
                    )
    
    # Status displays
    with gr.Row():
        auth_status = gr.Textbox(
            label="Authentication Status",
            value="No user selected",
            interactive=False
        )
    
    with gr.Row():
        config_status = gr.HTML(f"""
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <h4>🔧 Configuration Status</h4>
            <p>
                <strong>Demo Mode:</strong> {'✅ Available' if hybrid_manager.demo_enabled else '❌ Disabled'}<br>
                <strong>Real Auth:</strong> {'✅ Configured' if hybrid_manager.real_auth_enabled else '❌ Not Configured'}<br>
                <strong>Total Users:</strong> {len([u for u in ENHANCED_DEMO_USERS.values() if u.auth_type == 'demo'])} Demo + {'1 Real' if hybrid_manager.real_auth_enabled else '0 Real'}
            </p>
        </div>
        """)
    
    return {
        "hybrid_manager": hybrid_manager,
        "demo_buttons": demo_buttons,
        "real_auth_btn": real_auth_btn,
        "auth_status": auth_status
    }

def setup_authentication_handlers(components):
    """Setup event handlers for authentication"""
    
    def handle_demo_auth(user_key):
        """Handle demo user authentication"""
        result = components["hybrid_manager"].authenticate_user(user_key)
        
        if result["success"]:
            return (
                result["message"],
                gr.update(visible=True),  # Show main interface
                []  # Clear chat history
            )
        else:
            return (
                f"❌ {result['message']}",
                gr.update(visible=False),
                []
            )
    
    def handle_real_auth():
        """Handle real Dexcom authentication"""
        result = components["hybrid_manager"].authenticate_user("real_user")
        
        if result["success"]:
            return (
                f"✅ {result['message']} - Browser will open for OAuth",
                gr.update(visible=True),
                []
            )
        else:
            return (
                f"❌ {result['message']}",
                gr.update(visible=False),
                []
            )
    
    return handle_demo_auth, handle_real_auth

# Integration with your existing main.py
def integrate_with_existing_app():
    """Integration guide for your existing application"""
    
    integration_code = '''
# Add this to your main.py imports
from hybrid_dexcom_integration import HybridDexcomManager, ENHANCED_DEMO_USERS

class GlucoBuddyApp:
    def __init__(self):
        # Replace your existing initialization
        self.hybrid_manager = HybridDexcomManager()
        self.data_manager = UnifiedDataManager()
        self.mistral_chat = GlucoBuddyMistralChat()
        
        # UI state
        self.chat_history = []
    
    def select_user(self, user_key: str) -> Tuple[str, str]:
        """Enhanced user selection with hybrid auth"""
        try:
            # Use hybrid authentication
            auth_result = self.hybrid_manager.authenticate_user(user_key)
            
            if not auth_result['success']:
                return f"❌ {auth_result['message']}", gr.update(visible=False)
            
            # Load data based on auth type
            if auth_result['auth_type'] == 'demo':
                # Use mock data
                user = auth_result['user']
                data = auth_result['data']
                
                # Convert to format expected by UnifiedDataManager
                load_result = self.data_manager.load_mock_data(user, data)
            else:
                # Use real data
                user = auth_result['user']
                data = auth_result['data']
                
                # Convert to format expected by UnifiedDataManager
                load_result = self.data_manager.load_real_data(user, data)
            
            if load_result['success']:
                self._sync_chat_with_data_manager()
                self.chat_history = []
                self.mistral_chat.clear_conversation()
                
                return (
                    f"Connected: {user.name} ({auth_result['auth_type'].upper()}) - Click 'Load Data' to begin",
                    gr.update(visible=True)
                )
            else:
                return f"❌ Data loading failed: {load_result['message']}", gr.update(visible=False)
                
        except Exception as e:
            logger.error(f"User selection failed: {e}")
            return f"❌ Selection failed: {e}", gr.update(visible=False)

# Update your user buttons in create_interface()
def create_enhanced_interface():
    # Replace the user selection section with:
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👥 Select User")
            gr.Markdown("Choose demo users (instant) or real Dexcom authentication")
            
            # Demo users
            with gr.Row():
                sarah_btn = gr.Button("🎭 Sarah (Demo)\\nG7 Mobile", variant="secondary")
                marcus_btn = gr.Button("🎭 Marcus (Demo)\\nONE+ Mobile", variant="secondary") 
                jennifer_btn = gr.Button("🎭 Jennifer (Demo)\\nG6 Mobile", variant="secondary")
                robert_btn = gr.Button("🎭 Robert (Demo)\\nG6 Receiver", variant="secondary")
            
            # Real auth
            with gr.Row():
                real_auth_btn = gr.Button(
                    "🔐 REAL DEXCOM USER\\n(OAuth Authentication)",
                    variant="primary",
                    size="lg"
                )
    
    # Connect handlers:
    sarah_btn.click(lambda: app.select_user("sarah_demo"), outputs=[connection_status, main_interface, chatbot])
    marcus_btn.click(lambda: app.select_user("marcus_demo"), outputs=[connection_status, main_interface, chatbot])
    jennifer_btn.click(lambda: app.select_user("jennifer_demo"), outputs=[connection_status, main_interface, chatbot])
    robert_btn.click(lambda: app.select_user("robert_demo"), outputs=[connection_status, main_interface, chatbot])
    real_auth_btn.click(lambda: app.select_user("real_user"), outputs=[connection_status, main_interface, chatbot])
    '''
    
    return integration_code

if __name__ == "__main__":
    print("🔧 Hybrid Dexcom Integration - Demo + Real Authentication")
    print("=" * 60)
    
    # Test the hybrid manager
    manager = HybridDexcomManager()
    
    print("📊 Configuration Status:")
    print(f"  Demo Mode: {'✅' if manager.demo_enabled else '❌'}")
    print(f"  Real Auth: {'✅' if manager.real_auth_enabled else '❌'}")
    
    print(f"\n👥 Available Users:")
    for key, user in ENHANCED_DEMO_USERS.items():
        auth_icon = "🎭" if user.auth_type == "demo" else "🔐"
        available = "✅" if user.auth_type == "demo" or manager.real_auth_enabled else "❌"
        print(f"  {available} {auth_icon} {user.name}")
    
    print(f"\n💡 Integration Guide:")
    print("  1. Import: from hybrid_dexcom_integration import HybridDexcomManager")
    print("  2. Replace user selection in your main.py")
    print("  3. Update button handlers to use hybrid authentication")
    print("  4. Your existing UnifiedDataManager works with both data types!")
    
    print(f"\n🚀 Benefits:")
    print("  ✅ Keep all 4 demo users for easy demos")
    print("  ✅ Add real authentication when needed")
    print("  ✅ Seamless switching between demo and real")
    print("  ✅ No changes needed to existing chat/UI logic")
    print("  ✅ Progressive enhancement - works with/without real auth")