"""
GlucoBuddy - Dexcom API Integration Functions
This module contains all the functions for interacting with the Dexcom Sandbox API
and processing glucose data for AI insights.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DemoUser:
    """Demo user configuration for sandbox testing"""
    name: str
    device_type: str
    username: str
    description: str
    age: int = 30
    diabetes_type: str = "Type 1"
    years_with_diabetes: int = 5
    typical_glucose_pattern: str = "normal"


# Demo users based on Dexcom Sandbox documentation
DEMO_USERS = {
    "sarah_g7": DemoUser(
        name="Sarah Thompson",
        age=32,
        device_type="G7 Mobile App",
        username="User7",
        description="Active professional with Type 1 diabetes, uses G7 CGM with smartphone integration",
        diabetes_type="Type 1",
        years_with_diabetes=8,
        typical_glucose_pattern="stable_with_meal_spikes"
    ),
    "marcus_one": DemoUser(
        name="Marcus Rodriguez",
        age=45,
        device_type="ONE+ Mobile App", 
        username="User8",
        description="Father of two with Type 2 diabetes, manages with Dexcom ONE+ and lifestyle changes",
        diabetes_type="Type 2",
        years_with_diabetes=3,
        typical_glucose_pattern="moderate_variability"
    ),
    "jennifer_g6": DemoUser(
        name="Jennifer Chen",
        age=28,
        device_type="G6 Mobile App",
        username="User6",
        description="Graduate student with Type 1 diabetes, tech-savvy G6 user with active lifestyle",
        diabetes_type="Type 1",
        years_with_diabetes=12,
        typical_glucose_pattern="exercise_related_lows"
    ),
    "robert_receiver": DemoUser(
        name="Robert Williams",
        age=67,
        device_type="G6 Touchscreen Receiver",
        username="User4", 
        description="Retired teacher with Type 2 diabetes, prefers dedicated receiver device",
        diabetes_type="Type 2",
        years_with_diabetes=15,
        typical_glucose_pattern="dawn_phenomenon"
    )
}

# Dexcom API Configuration
SANDBOX_BASE_URL = "https://sandbox-api.dexcom.com"
CLIENT_ID = "your_client_id_here"  # Replace with your actual client ID
CLIENT_SECRET = "your_client_secret_here"  # Replace with your actual client secret
REDIRECT_URI = "http://localhost:8080/callback"  # Adjust as needed


class DexcomAPI:
    """Handles all Dexcom API interactions"""
    
    def __init__(self):
        self.base_url = SANDBOX_BASE_URL
        self.access_token = None
        self.refresh_token = None
        
    def get_authorization_url(self, state: str = None) -> str:
        """Generate the OAuth authorization URL for user login"""
        params = {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": "offline_access"
        }
        if state:
            params["state"] = state
            
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}/v2/oauth2/login?{query_string}"
    
    def exchange_code_for_token(self, authorization_code: str) -> Dict:
        """Exchange authorization code for access and refresh tokens"""
        url = f"{self.base_url}/v2/oauth2/token"
        
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": authorization_code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        try:
            response = requests.post(url, data=data, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            
            return token_data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to exchange authorization code: {str(e)}")
    
    def simulate_demo_login(self, demo_user_key: str) -> str:
        """Simulate getting an access token for demo users in sandbox"""
        # In a real implementation, this would go through the OAuth flow
        # For demo purposes, we'll simulate having tokens for each user
        demo_tokens = {
            "sarah_g7": "demo_token_user7_g7_mobile",
            "marcus_one": "demo_token_user8_oneplus_mobile", 
            "jennifer_g6": "demo_token_user6_g6_mobile",
            "robert_receiver": "demo_token_user4_g6_receiver"
        }
        
        if demo_user_key not in demo_tokens:
            raise ValueError(f"Invalid demo user: {demo_user_key}")
            
        self.access_token = demo_tokens[demo_user_key]
        return self.access_token
    
    def refresh_access_token(self) -> Dict:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            raise Exception("No refresh token available")
            
        url = f"{self.base_url}/v2/oauth2/token"
        
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token"
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        try:
            response = requests.post(url, data=data, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            
            return token_data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to refresh token: {str(e)}")
    
    def get_data_range(self) -> Dict:
        """Get the available data range for the authenticated user"""
        if not self.access_token:
            raise Exception("No access token available")
            
        url = f"{self.base_url}/v2/users/self/dataRange"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get data range: {str(e)}")
    
    def get_egv_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get Estimated Glucose Values (EGV) data"""
        if not self.access_token:
            raise Exception("No access token available")
            
        url = f"{self.base_url}/v2/users/self/egvs"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
            
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("egvs", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get EGV data: {str(e)}")
    
    def get_events_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get events data (meals, insulin, etc.)"""
        if not self.access_token:
            raise Exception("No access token available")
            
        url = f"{self.base_url}/v2/users/self/events"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
            
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("events", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get events data: {str(e)}")


class GlucoseAnalyzer:
    """Analyzes glucose data and generates insights"""
    
    @staticmethod
    def process_egv_data(egv_data: List[Dict]) -> pd.DataFrame:
        """Convert EGV data to pandas DataFrame for analysis"""
        if not egv_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(egv_data)
        df['systemTime'] = pd.to_datetime(df['systemTime'])
        df['displayTime'] = pd.to_datetime(df['displayTime'])
        
        # Convert glucose values to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df.sort_values('systemTime')
    
    @staticmethod
    def calculate_basic_stats(df: pd.DataFrame) -> Dict:
        """Calculate basic glucose statistics"""
        if df.empty:
            return {}
            
        glucose_values = df['value'].dropna()
        
        return {
            "average_glucose": glucose_values.mean(),
            "min_glucose": glucose_values.min(),
            "max_glucose": glucose_values.max(),
            "std_glucose": glucose_values.std(),
            "time_in_range_70_180": len(glucose_values[(glucose_values >= 70) & (glucose_values <= 180)]) / len(glucose_values) * 100,
            "time_below_70": len(glucose_values[glucose_values < 70]) / len(glucose_values) * 100,
            "time_above_180": len(glucose_values[glucose_values > 180]) / len(glucose_values) * 100,
            "total_readings": len(glucose_values)
        }
    
    @staticmethod
    def identify_patterns(df: pd.DataFrame) -> Dict:
        """Identify glucose patterns and trends"""
        if df.empty or len(df) < 10:
            return {"patterns": "Insufficient data for pattern analysis"}
            
        patterns = []
        
        # Daily patterns
        df['hour'] = df['systemTime'].dt.hour
        hourly_avg = df.groupby('hour')['value'].mean()
        
        # Find peak and low times
        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()
        
        patterns.append(f"Glucose typically peaks around {peak_hour}:00")
        patterns.append(f"Glucose is typically lowest around {low_hour}:00")
        
        # Variability patterns
        glucose_std = df['value'].std()
        if glucose_std > 50:
            patterns.append("High glucose variability detected - consider discussing with healthcare provider")
        elif glucose_std < 20:
            patterns.append("Good glucose stability observed")
        
        # Trend analysis
        recent_data = df.tail(20)  # Last 20 readings
        if len(recent_data) >= 10:
            trend_slope = (recent_data['value'].iloc[-1] - recent_data['value'].iloc[0]) / len(recent_data)
            if trend_slope > 2:
                patterns.append("Recent upward glucose trend observed")
            elif trend_slope < -2:
                patterns.append("Recent downward glucose trend observed")
            else:
                patterns.append("Glucose levels relatively stable recently")
        
        return {"patterns": patterns}


class ClaudeMCPClient:
    """Client for interacting with Claude via MCP"""
    
    def __init__(self):
        try:
            import anthropic
            from config import CLAUDE_CONFIG
            self.client = anthropic.Anthropic(api_key=CLAUDE_CONFIG["API_KEY"])
            self.model = CLAUDE_CONFIG["MODEL"]
            self.max_tokens = CLAUDE_CONFIG["MAX_TOKENS"]
            self.temperature = CLAUDE_CONFIG["TEMPERATURE"]
            self.available = True
        except (ImportError, Exception) as e:
            print(f"Claude MCP not available: {e}")
            self.available = False
    
    def generate_insights(self, glucose_stats: Dict, patterns: Dict, user_info, events_data: List = None) -> str:
        """Generate AI insights using Claude MCP"""
        
        if not self.available:
            return self._generate_fallback_insights(glucose_stats, patterns, user_info)
        
        # Prepare comprehensive context for Claude
        context = self._prepare_analysis_context(glucose_stats, patterns, user_info, events_data)
        
        system_prompt = """You are GlucoBuddy, an AI assistant specialized in analyzing continuous glucose monitoring (CGM) data. 
        You provide personalized, actionable insights while being supportive and encouraging. 
        Always remind users to consult with their healthcare providers for medical decisions.
        
        Analyze the provided glucose data and generate insights focusing on:
        - Pattern recognition and trends
        - Time-in-range performance
        - Risk identification and prevention
        - Personalized recommendations
        - Device-specific optimization tips
        
        Be empathetic, clear, evidence-based, and use emojis to make the analysis engaging."""
        
        user_prompt = f"""Please analyze this CGM data and provide comprehensive insights:

        {context}
        
        Generate a detailed analysis with actionable recommendations formatted in markdown."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return self._generate_fallback_insights(glucose_stats, patterns, user_info)
    
    def _prepare_analysis_context(self, glucose_stats: Dict, patterns: Dict, user_info, events_data: List = None) -> str:
        """Prepare comprehensive context for Claude analysis"""
        
        context = f"""
        ## Patient Profile
        **Name**: {user_info.name}
        **Device**: {user_info.device_type}
        **Diabetes Type**: {getattr(user_info, 'diabetes_type', 'Not specified')}
        **Years with Diabetes**: {getattr(user_info, 'years_with_diabetes', 'Not specified')}
        
        ## Glucose Statistics Summary
        - **Average Glucose**: {glucose_stats.get('average_glucose', 0):.1f} mg/dL
        - **Glucose Range**: {glucose_stats.get('min_glucose', 0):.1f} - {glucose_stats.get('max_glucose', 0):.1f} mg/dL
        - **Standard Deviation**: {glucose_stats.get('std_glucose', 0):.1f} mg/dL
        - **Time in Range (70-180 mg/dL)**: {glucose_stats.get('time_in_range_70_180', 0):.1f}%
        - **Time Below 70 mg/dL**: {glucose_stats.get('time_below_70', 0):.1f}%
        - **Time Above 180 mg/dL**: {glucose_stats.get('time_above_180', 0):.1f}%
        - **Total Readings**: {glucose_stats.get('total_readings', 0)}
        
        ## Identified Patterns
        {self._format_patterns(patterns)}
        
        ## Clinical Context
        - Target Time in Range: 70% or higher is excellent, 50-70% is good
        - Hypoglycemia concern: <4% time below 70 mg/dL is ideal
        - Glucose variability: Lower standard deviation indicates better stability
        """
        
        if events_data:
            context += f"\n## Events Data\n{self._format_events_data(events_data)}"
        
        return context
    
    def _format_patterns(self, patterns: Dict) -> str:
        """Format patterns data for context"""
        if isinstance(patterns.get('patterns'), list):
            return "\n".join([f"- {pattern}" for pattern in patterns['patterns']])
        else:
            return f"- {patterns.get('patterns', 'No specific patterns identified')}"
    
    def _format_events_data(self, events_data: List) -> str:
        """Format events data for context"""
        if not events_data:
            return "No events data available"
        
        formatted = []
        for event in events_data[:10]:  # Limit to recent events
            event_type = event.get('eventType', 'Unknown')
            event_time = event.get('systemTime', 'Unknown time')
            formatted.append(f"- {event_type} at {event_time}")
        
        return "\n".join(formatted)
    
    def _generate_fallback_insights(self, glucose_stats: Dict, patterns: Dict, user_info) -> str:
        """Generate fallback insights when Claude MCP is not available"""
        
        return f"""
        ## 🩺 Glucose Management Insights for {user_info.name}
        
        ### Overall Assessment
        Based on your {user_info.device_type} data, here's what I've observed:

        **Time in Range Performance**: {glucose_stats.get('time_in_range_70_180', 0):.1f}%
        - Target range is typically 70-180 mg/dL
        - {'✅ Excellent control!' if glucose_stats.get('time_in_range_70_180', 0) > 80 else '⚠️ Room for improvement' if glucose_stats.get('time_in_range_70_180', 0) > 60 else '🔴 Needs attention'}

        ### Key Observations
        
        **Glucose Stability**: 
        - Standard deviation: {glucose_stats.get('std_glucose', 0):.1f} mg/dL
        - {'Good stability' if glucose_stats.get('std_glucose', 0) < 30 else 'Moderate variability' if glucose_stats.get('std_glucose', 0) < 50 else 'High variability - consider discussing with your healthcare team'}

        **Hypoglycemia Risk**: 
        - Time below 70 mg/dL: {glucose_stats.get('time_below_70', 0):.1f}%
        - {'Low risk' if glucose_stats.get('time_below_70', 0) < 4 else 'Moderate risk - monitor closely' if glucose_stats.get('time_below_70', 0) < 10 else 'Higher risk - consult your healthcare provider'}

        ### Personalized Recommendations

        1. **Pattern-Based Suggestions**:
           {chr(10).join(['   - ' + p for p in (patterns.get('patterns', []) if isinstance(patterns.get('patterns'), list) else [str(patterns.get('patterns', ''))])])}

        2. **Device-Specific Tips**:
           {'Consider using smartphone alerts for better real-time monitoring' if 'Mobile App' in user_info.device_type else 'Regular data downloads recommended for comprehensive tracking'}

        3. **Lifestyle Considerations**:
           - Monitor glucose before and after meals
           - Track exercise impact on glucose levels
           - Consider stress management techniques

        ### Next Steps
        - Review this data with your healthcare provider
        - Set up alerts for out-of-range values
        - Consider keeping a food and activity log

        *This analysis is based on your CGM data and is for informational purposes only. Always consult with your healthcare provider for medical decisions.*
        
        ---
        *Note: Enhanced AI insights unavailable - using basic analysis mode.*
        """


def generate_ai_insights(glucose_stats: Dict, patterns: Dict, user_info: DemoUser, events_data: List = None) -> str:
    """Generate AI-powered insights using Claude MCP"""
    claude_client = ClaudeMCPClient()
    return claude_client.generate_insights(glucose_stats, patterns, user_info, events_data)


def format_glucose_data_for_display(df: pd.DataFrame) -> str:
    """Format glucose data for display in the interface"""
    if df.empty:
        return "No glucose data available"
    
    # Get recent readings
    recent_data = df.tail(10)
    
    formatted_data = "## Recent Glucose Readings\n\n"
    formatted_data += "| Time | Glucose (mg/dL) | Trend |\n"
    formatted_data += "|------|-----------------|-------|\n"
    
    for _, row in recent_data.iterrows():
        time_str = row['displayTime'].strftime("%m/%d %H:%M")
        glucose = row['value']
        trend = row.get('trend', 'N/A')
        
        # Convert trend to arrow
        trend_arrow = {
            'flat': '→',
            'fortyFiveUp': '↗',
            'singleUp': '↑',
            'doubleUp': '⬆',
            'fortyFiveDown': '↘',
            'singleDown': '↓',
            'doubleDown': '⬇'
        }.get(trend, '→')
        
        formatted_data += f"| {time_str} | {glucose:.0f} | {trend_arrow} |\n"
    
    return formatted_data