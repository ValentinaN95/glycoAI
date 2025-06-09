#!/usr/bin/env python3
"""
Dexcom Client for glucose data analysis
"""

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import asdict

"""
Dexcom API Integration Functions
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
import base64
import secrets
import hashlib
import requests
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DemoUser:
    """Demo user configuration for sandbox testing"""
    name: str
    device_type: str
    username: str
    password: str
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
        username="sandboxuser7@dexcom.com",
        password="Dexcom123!",
        description="Active professional with Type 1 diabetes, uses G7 CGM with smartphone integration",
        diabetes_type="Type 1",
        years_with_diabetes=8,
        typical_glucose_pattern="stable_with_meal_spikes"
    ),
    "marcus_one": DemoUser(
        name="Marcus Rodriguez",
        age=45,
        device_type="ONE+ Mobile App",
        username="sandboxuser8@dexcom.com",
        password="Dexcom123!",
        description="Father of two with Type 2 diabetes, manages with Dexcom ONE+ and lifestyle changes",
        diabetes_type="Type 2",
        years_with_diabetes=3,
        typical_glucose_pattern="moderate_variability"
    ),
    "jennifer_g6": DemoUser(
        name="Jennifer Chen",
        age=28,
        device_type="G6 Mobile App",
        username="sandboxuser6@dexcom.com",
        password="Dexcom123!",
        description="Graduate student with Type 1 diabetes, tech-savvy G6 user with active lifestyle",
        diabetes_type="Type 1",
        years_with_diabetes=12,
        typical_glucose_pattern="exercise_related_lows"
    ),
    "robert_receiver": DemoUser(
        name="Robert Williams",
        age=67,
        device_type="G6 Touchscreen Receiver",
        username="sandboxuser4@dexcom.com",
        password="Dexcom123!",
        description="Retired teacher with Type 2 diabetes, prefers dedicated receiver device",
        diabetes_type="Type 2",
        years_with_diabetes=15,
        typical_glucose_pattern="dawn_phenomenon"
    )
}

# Dexcom API Configuration
SANDBOX_BASE_URL = "https://sandbox-api.dexcom.com"
CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
REDIRECT_URI = "http://localhost:8080/callback"

class DexcomAPI:
    """Handles all Dexcom API interactions with proper OAuth flow"""

    def __init__(self):
        self.base_url = SANDBOX_BASE_URL
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

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

            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            return token_data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to exchange authorization code: {str(e)}")

    def simulate_demo_login(self, demo_user_key: str) -> str:
        """Simulate OAuth flow for demo users using sandbox credentials"""
        if demo_user_key not in DEMO_USERS:
            raise ValueError(f"Invalid demo user: {demo_user_key}")

        user = DEMO_USERS[demo_user_key]

        try:
            auth_code = self._simulate_sandbox_login(user.username, user.password)

            if auth_code:
                token_data = self.exchange_code_for_token(auth_code)
                logger.info(f"Successfully obtained tokens for {user.name}")
                return self.access_token
            else:
                return self._direct_sandbox_token_request(user.username, user.password)

        except Exception as e:
            logger.warning(f"OAuth flow failed, trying direct sandbox authentication: {e}")
            return self._direct_sandbox_token_request(user.username, user.password)

    def _simulate_sandbox_login(self, username: str, password: str) -> Optional[str]:
        """Simulate the sandbox login process to get authorization code"""
        try:
            auth_string = f"{username}:{password}:{datetime.now().strftime('%Y%m%d')}"
            auth_code = base64.b64encode(auth_string.encode()).decode()[:32]
            return auth_code

        except Exception as e:
            logger.error(f"Failed to simulate sandbox login: {e}")
            return None

    def _direct_sandbox_token_request(self, username: str, password: str) -> str:
        """Direct token request for sandbox environment"""
        url = f"{self.base_url}/v2/oauth2/token"
        credentials = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "grant_type": "password",
            "username": username,
            "password": password,
            "scope": "offline_access"
        }

        try:
            response = requests.post(url, data=data, headers=headers)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get("access_token")
                self.refresh_token = token_data.get("refresh_token")

                expires_in = token_data.get("expires_in", 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

                logger.info(f"Successfully obtained sandbox token for {username}")
                return self.access_token
            else:
                sandbox_token = self._generate_sandbox_demo_token(username)
                self.access_token = sandbox_token
                self.token_expires_at = datetime.now() + timedelta(hours=1)

                logger.info(f"Using demo token for sandbox user {username}")
                return sandbox_token

        except Exception as e:
            logger.error(f"Direct token request failed: {e}")
            sandbox_token = self._generate_sandbox_demo_token(username)
            self.access_token = sandbox_token
            self.token_expires_at = datetime.now() + timedelta(hours=1)
            return sandbox_token

    def _generate_sandbox_demo_token(self, username: str) -> str:
        """Generate a realistic-looking demo token for sandbox"""
        token_data = f"{username}:{datetime.now().strftime('%Y%m%d')}:{CLIENT_ID}"
        token_hash = hashlib.sha256(token_data.encode()).hexdigest()
        return f"sandbox_token_{token_hash[:16]}"

    def _is_token_expired(self) -> bool:
        """Check if the current token is expired"""
        if not self.token_expires_at:
            return True
        return datetime.now() >= self.token_expires_at

    def _ensure_valid_token(self):
        """Ensure we have a valid, non-expired token"""
        if not self.access_token or self._is_token_expired():
            if self.refresh_token:
                try:
                    self.refresh_access_token()
                except:
                    raise Exception("Token expired and refresh failed. Please re-authenticate.")
            else:
                raise Exception("No valid token available. Please authenticate first.")

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
            if "refresh_token" in token_data:
                self.refresh_token = token_data.get("refresh_token")

            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            return token_data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to refresh token: {str(e)}")

    def get_data_range(self) -> Dict:
        """Get the available data range for the authenticated user"""
        self._ensure_valid_token()

        url = f"{self.base_url}/v2/users/self/dataRange"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 401:
                if self.refresh_token:
                    self.refresh_access_token()
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "egvStart": (datetime.now() - timedelta(days=30)).isoformat(),
                    "egvEnd": datetime.now().isoformat(),
                    "eventStart": (datetime.now() - timedelta(days=30)).isoformat(),
                    "eventEnd": datetime.now().isoformat()
                }

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get data range from API: {e}, using demo range")
            return {
                "egvStart": (datetime.now() - timedelta(days=30)).isoformat(),
                "egvEnd": datetime.now().isoformat(),
                "eventStart": (datetime.now() - timedelta(days=30)).isoformat(),
                "eventEnd": datetime.now().isoformat()
            }

    def get_egv_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get Estimated Glucose Values (EGV) data"""
        self._ensure_valid_token()

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

            if response.status_code == 401:
                if self.refresh_token:
                    self.refresh_access_token()
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                return data.get("egvs", [])
            else:
                logger.warning(f"API returned status {response.status_code}, generating demo data")
                return []

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get EGV data from API: {e}, will use demo data")
            return []

    def get_events_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get events data (meals, insulin, etc.)"""
        self._ensure_valid_token()

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

            if response.status_code == 401:
                if self.refresh_token:
                    self.refresh_access_token()
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                return data.get("events", [])
            else:
                logger.warning(f"Events API returned status {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get events data: {e}")
            return []

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

        df['hour'] = df['systemTime'].dt.hour
        hourly_avg = df.groupby('hour')['value'].mean()

        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()

        patterns.append(f"Glucose typically peaks around {peak_hour}:00")
        patterns.append(f"Glucose is typically lowest around {low_hour}:00")

        glucose_std = df['value'].std()
        if glucose_std > 50:
            patterns.append("High glucose variability detected - consider discussing with healthcare provider")
        elif glucose_std < 20:
            patterns.append("Good glucose stability observed")

        recent_data = df.tail(20)
        if len(recent_data) >= 10:
            trend_slope = (recent_data['value'].iloc[-1] - recent_data['value'].iloc[0]) / len(recent_data)
            if trend_slope > 2:
                patterns.append("Recent upward glucose trend observed")
            elif trend_slope < -2:
                patterns.append("Recent downward glucose trend observed")
            else:
                patterns.append("Glucose levels relatively stable recently")

        return {"patterns": patterns}
def format_glucose_data_for_display(df: pd.DataFrame) -> str:
    """Format glucose data for display in the interface"""
    if df.empty:
        return "No glucose data available"

    recent_data = df.tail(10)

    formatted_data = "## Recent Glucose Readings\n\n"
    formatted_data += "| Time | Glucose (mg/dL) | Trend |\n"
    formatted_data += "|------|-----------------|-------|\n"

    for _, row in recent_data.iterrows():
        time_str = row['displayTime'].strftime("%m/%d %H:%M")
        glucose = row['value']
        trend = row.get('trend', 'N/A')

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