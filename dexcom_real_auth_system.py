#!/usr/bin/env python3
"""
Dexcom Real Authentication System
Uses your actual CLIENT_ID and CLIENT_SECRET from developer.dexcom.com
"""

import requests
import json
import base64
import secrets
import hashlib
import urllib.parse
import webbrowser
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔑 YOUR REAL DEXCOM CREDENTIALS (replace with your actual values)
CLIENT_ID = "YOUR_REAL_CLIENT_ID_HERE"  # Replace with your actual client ID
CLIENT_SECRET = "YOUR_REAL_CLIENT_SECRET_HERE"  # Replace with your actual client secret
REDIRECT_URI = "http://localhost:8080/callback"

# Dexcom API URLs
SANDBOX_BASE_URL = "https://sandbox-api.dexcom.com"
PRODUCTION_BASE_URL = "https://api.dexcom.com"

class OAuth2CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth2 callback"""
    
    def do_GET(self):
        """Handle GET request for OAuth callback"""
        if self.path.startswith('/callback'):
            # Parse the authorization code from the URL
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                # Store the authorization code in the server
                self.server.auth_code = query_params['code'][0]
                self.server.auth_error = None
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                success_html = """
                <html>
                <head><title>Dexcom Authorization Successful</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h2 style="color: green;">✅ Authorization Successful!</h2>
                    <p>You have successfully authorized the application to access your Dexcom data.</p>
                    <p>You can close this window and return to the application.</p>
                    <script>
                        setTimeout(function(){
                            window.close();
                        }, 3000);
                    </script>
                </body>
                </html>
                """
                self.wfile.write(success_html.encode())
                
            elif 'error' in query_params:
                # Handle authorization error
                error = query_params.get('error', ['Unknown error'])[0]
                error_description = query_params.get('error_description', [''])[0]
                
                self.server.auth_code = None
                self.server.auth_error = f"{error}: {error_description}"
                
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = f"""
                <html>
                <head><title>Dexcom Authorization Failed</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h2 style="color: red;">❌ Authorization Failed</h2>
                    <p><strong>Error:</strong> {error}</p>
                    <p><strong>Description:</strong> {error_description}</p>
                    <p>Please try again or contact support if the problem persists.</p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
            else:
                # Unexpected callback
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h2>Invalid callback</h2></body></html>")
        else:
            # 404 for other paths
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

class DexcomRealAPI:
    """Real Dexcom API client using your actual credentials"""
    
    def __init__(self, client_id: str = CLIENT_ID, client_secret: str = CLIENT_SECRET, 
                 environment: str = "sandbox"):
        """
        Initialize Dexcom API client
        
        Args:
            client_id: Your real Dexcom client ID
            client_secret: Your real Dexcom client secret  
            environment: "sandbox" or "production"
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = REDIRECT_URI
        
        if environment == "sandbox":
            self.base_url = SANDBOX_BASE_URL
        else:
            self.base_url = PRODUCTION_BASE_URL
            
        self.environment = environment
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        
        # Validate credentials
        if not client_id or client_id == "YOUR_REAL_CLIENT_ID_HERE":
            raise ValueError("Please set your real CLIENT_ID")
        if not client_secret or client_secret == "YOUR_REAL_CLIENT_SECRET_HERE":
            raise ValueError("Please set your real CLIENT_SECRET")
    
    def generate_auth_url(self, state: str = None) -> str:
        """Generate OAuth authorization URL"""
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'offline_access',
            'state': state
        }
        
        query_string = urllib.parse.urlencode(params)
        auth_url = f"{self.base_url}/v2/oauth2/login?{query_string}"
        
        logger.info(f"Generated authorization URL for {self.environment} environment")
        return auth_url
    
    def start_oauth_flow(self) -> bool:
        """Start the complete OAuth flow with browser"""
        print(f"\n🔐 Starting Dexcom OAuth Authentication")
        print(f"🌐 Environment: {self.environment}")
        print(f"🔑 Client ID: {self.client_id[:8]}...")
        
        try:
            # Generate authorization URL
            state = secrets.token_urlsafe(32)
            auth_url = self.generate_auth_url(state)
            
            # Start local callback server
            server = HTTPServer(('localhost', 8080), OAuth2CallbackHandler)
            server.timeout = 120  # 2 minute timeout
            server.auth_code = None
            server.auth_error = None
            
            print(f"🌐 Opening browser for authorization...")
            print(f"📋 URL: {auth_url}")
            print(f"⏳ Waiting for authorization (timeout: 2 minutes)...")
            
            # Open browser
            webbrowser.open(auth_url)
            
            # Wait for callback
            start_time = time.time()
            while time.time() - start_time < 120:  # 2 minute timeout
                server.handle_request()
                if server.auth_code or server.auth_error:
                    break
            
            if server.auth_error:
                print(f"❌ Authorization failed: {server.auth_error}")
                return False
            
            if not server.auth_code:
                print(f"❌ Authorization timeout - no response received")
                return False
            
            print(f"✅ Authorization code received!")
            
            # Exchange code for tokens
            success = self.exchange_code_for_tokens(server.auth_code)
            
            if success:
                print(f"🎉 Authentication successful!")
                print(f"📊 Access token obtained")
                print(f"⏰ Token expires: {self.token_expires_at}")
                return True
            else:
                print(f"❌ Token exchange failed")
                return False
                
        except Exception as e:
            logger.error(f"OAuth flow error: {e}")
            print(f"❌ OAuth flow error: {e}")
            return False
    
    def exchange_code_for_tokens(self, authorization_code: str) -> bool:
        """Exchange authorization code for access and refresh tokens"""
        url = f"{self.base_url}/v2/oauth2/token"
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': authorization_code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        try:
            logger.info("Exchanging authorization code for tokens...")
            response = requests.post(url, data=data, headers=headers)
            
            logger.info(f"Token exchange response: {response.status_code}")
            
            if response.status_code == 200:
                token_data = response.json()
                
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                logger.info("Successfully obtained access and refresh tokens")
                return True
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during token exchange: {e}")
            return False
    
    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False
        
        url = f"{self.base_url}/v2/oauth2/token"
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        try:
            response = requests.post(url, data=data, headers=headers)
            
            if response.status_code == 200:
                token_data = response.json()
                
                self.access_token = token_data.get('access_token')
                # Note: Some OAuth providers issue new refresh tokens
                if 'refresh_token' in token_data:
                    self.refresh_token = token_data.get('refresh_token')
                
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                logger.info("Access token refreshed successfully")
                return True
            else:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during token refresh: {e}")
            return False
    
    def is_token_valid(self) -> bool:
        """Check if current access token is valid and not expired"""
        if not self.access_token:
            return False
        
        if not self.token_expires_at:
            return True  # Assume valid if no expiry set
        
        # Consider token expired if it expires within next 5 minutes
        return datetime.now() < (self.token_expires_at - timedelta(minutes=5))
    
    def ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token, refresh if needed"""
        if not self.is_token_valid():
            logger.info("Token expired or invalid, attempting refresh...")
            if self.refresh_token:
                return self.refresh_access_token()
            else:
                logger.error("No refresh token available, re-authentication required")
                return False
        return True
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers with valid authorization token"""
        if not self.ensure_valid_token():
            raise Exception("No valid access token available. Please authenticate first.")
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
    
    def get_data_range(self) -> Dict:
        """Get available data range for authenticated user"""
        url = f"{self.base_url}/v2/users/self/dataRange"
        headers = self.get_auth_headers()
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Data range API error: {response.status_code} - {response.text}")
                raise Exception(f"Data range API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting data range: {e}")
            raise Exception(f"Network error getting data range: {e}")
    
    def get_egv_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get Estimated Glucose Values (EGV) data"""
        url = f"{self.base_url}/v2/users/self/egvs"
        headers = self.get_auth_headers()
        
        params = {}
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('egvs', [])
            else:
                logger.error(f"EGV API error: {response.status_code} - {response.text}")
                raise Exception(f"EGV API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting EGV data: {e}")
            raise Exception(f"Network error getting EGV data: {e}")
    
    def get_events_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get events data (meals, insulin, etc.)"""
        url = f"{self.base_url}/v2/users/self/events"
        headers = self.get_auth_headers()
        
        params = {}
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('events', [])
            else:
                logger.error(f"Events API error: {response.status_code} - {response.text}")
                raise Exception(f"Events API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting events data: {e}")
            raise Exception(f"Network error getting events data: {e}")

def test_real_dexcom_api():
    """Test the real Dexcom API with your credentials"""
    print("🧪 Testing Real Dexcom API Integration")
    print("=" * 60)
    
    try:
        # Initialize API with your real credentials
        api = DexcomRealAPI(environment="sandbox")
        
        # Start OAuth authentication
        print("\n🔐 Step 1: Authentication")
        auth_success = api.start_oauth_flow()
        
        if not auth_success:
            print("❌ Authentication failed - cannot proceed")
            return False
        
        # Test data range
        print("\n📅 Step 2: Getting Data Range")
        try:
            data_range = api.get_data_range()
            print(f"✅ Data range retrieved:")
            print(f"   EGV: {data_range.get('egvStart', 'N/A')} to {data_range.get('egvEnd', 'N/A')}")
            print(f"   Events: {data_range.get('eventStart', 'N/A')} to {data_range.get('eventEnd', 'N/A')}")
        except Exception as e:
            print(f"❌ Data range error: {e}")
        
        # Test glucose data
        print("\n📊 Step 3: Getting Glucose Data")
        try:
            # Get last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            egv_data = api.get_egv_data(
                start_date=start_time.isoformat(),
                end_date=end_time.isoformat()
            )
            
            print(f"✅ Retrieved {len(egv_data)} glucose readings")
            
            if egv_data:
                latest = egv_data[-1]
                print(f"   Latest: {latest['value']} mg/dL at {latest['displayTime']}")
                print(f"   Trend: {latest.get('trend', 'N/A')}")
        except Exception as e:
            print(f"❌ Glucose data error: {e}")
        
        # Test events data
        print("\n🍽️ Step 4: Getting Events Data")
        try:
            events_data = api.get_events_data(
                start_date=start_time.isoformat(),
                end_date=end_time.isoformat()
            )
            
            print(f"✅ Retrieved {len(events_data)} events")
            
            if events_data:
                carb_events = [e for e in events_data if e.get('eventType') == 'carbs']
                insulin_events = [e for e in events_data if e.get('eventType') == 'insulin']
                print(f"   Carb events: {len(carb_events)}")
                print(f"   Insulin events: {len(insulin_events)}")
        except Exception as e:
            print(f"❌ Events data error: {e}")
        
        print(f"\n🎉 Real Dexcom API integration completed!")
        return True
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print(f"💡 Please update CLIENT_ID and CLIENT_SECRET with your real credentials")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🔑 Dexcom Real API Authentication System")
    print("📋 Make sure to update CLIENT_ID and CLIENT_SECRET with your real values!")
    print()
    
    # Run the test
    test_real_dexcom_api()
    
    print(f"\n💡 Usage Example:")
    print(f"   api = DexcomRealAPI(environment='sandbox')")
    print(f"   api.start_oauth_flow()  # Opens browser for auth")
    print(f"   glucose_data = api.get_egv_data()")
    print(f"   events_data = api.get_events_data()")
    
    print(f"\n⚠️  Important Notes:")
    print(f"   • This uses the real Dexcom sandbox environment")
    print(f"   • You'll need to authenticate through the browser")
    print(f"   • Sandbox users are provided by Dexcom (like SandboxUser7)")
    print(f"   • Update environment='production' for real user data")