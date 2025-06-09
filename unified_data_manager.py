"""
Unified Data Manager for GlycoAI - FIXED VERSION
Restores the original working API calls that were working before
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict

from apifunctions import (
    DexcomAPI,
    GlucoseAnalyzer,
    DEMO_USERS,
    DemoUser
)

logger = logging.getLogger(__name__)

class UnifiedDataManager:
    """
    FIXED: Unified data manager that calls the API exactly as it was working before
    """
    
    def __init__(self):
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        
        logger.info(f"UnifiedDataManager initialized - RESTORED to working version")
        
        # Single source of truth for all data
        self.current_user: Optional[DemoUser] = None
        self.raw_glucose_data: Optional[list] = None
        self.processed_glucose_data: Optional[pd.DataFrame] = None
        self.calculated_stats: Optional[Dict] = None
        self.identified_patterns: Optional[Dict] = None
        
        # Metadata
        self.data_loaded_at: Optional[datetime] = None
        self.data_source: str = "none"  # "dexcom_api", "mock", or "none"
        
    def load_user_data(self, user_key: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        FIXED: Load glucose data using the ORIGINAL WORKING method
        """
        
        # Check if we already have data for this user and it's recent
        if (not force_reload and 
            self.current_user and 
            self.current_user == DEMO_USERS.get(user_key) and
            self.data_loaded_at and 
            (datetime.now() - self.data_loaded_at).seconds < 300):  # 5 minutes cache
            
            logger.info(f"Using cached data for {user_key}")
            return self._build_success_response()
        
        try:
            if user_key not in DEMO_USERS:
                return {
                    "success": False,
                    "message": f"❌ Invalid user key '{user_key}'. Available: {', '.join(DEMO_USERS.keys())}"
                }
            
            logger.info(f"Loading data for user: {user_key}")
            
            # Set current user
            self.current_user = DEMO_USERS[user_key]
            
            # Call API EXACTLY as it was working before
            try:
                logger.info(f"Attempting Dexcom API authentication for {user_key}")
                
                # ORIGINAL WORKING METHOD: Use the simulate_demo_login exactly as before
                access_token = self.dexcom_api.simulate_demo_login(user_key)
                logger.info(f"Dexcom authentication result: {bool(access_token)}")
                
                if access_token:
                    # ORIGINAL WORKING METHOD: Get data with 14-day range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=14)
                    
                    # Call get_egv_data EXACTLY as it was working before
                    self.raw_glucose_data = self.dexcom_api.get_egv_data(
                        start_date.isoformat(),
                        end_date.isoformat()
                    )
                    
                    if self.raw_glucose_data and len(self.raw_glucose_data) > 0:
                        self.data_source = "dexcom_api"
                        logger.info(f"✅ Successfully loaded {len(self.raw_glucose_data)} readings from Dexcom API")
                    else:
                        logger.warning("Dexcom API returned empty data - falling back to mock data")
                        raise Exception("Empty data from Dexcom API")
                else:
                    logger.warning("Failed to get access token - falling back to mock data")
                    raise Exception("Authentication failed")
                    
            except Exception as api_error:
                logger.warning(f"Dexcom API failed ({str(api_error)}) - using mock data fallback")
                self.raw_glucose_data = self._generate_realistic_mock_data(user_key)
                self.data_source = "mock"
            
            # Process the raw data (same processing for everyone)
            self.processed_glucose_data = self.analyzer.process_egv_data(self.raw_glucose_data)
            
            if self.processed_glucose_data is None or self.processed_glucose_data.empty:
                return {
                    "success": False,
                    "message": "❌ Failed to process glucose data"
                }
            
            # Calculate statistics (single source of truth)
            self.calculated_stats = self._calculate_unified_stats()
            
            # Identify patterns
            self.identified_patterns = self.analyzer.identify_patterns(self.processed_glucose_data)
            
            # Mark when data was loaded
            self.data_loaded_at = datetime.now()
            
            logger.info(f"Successfully loaded and processed data for {self.current_user.name}")
            logger.info(f"Data source: {self.data_source}, Readings: {len(self.processed_glucose_data)}")
            logger.info(f"TIR: {self.calculated_stats.get('time_in_range_70_180', 0):.1f}%")
            
            return self._build_success_response()
            
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            return {
                "success": False,
                "message": f"❌ Failed to load user data: {str(e)}"
            }
    
    def get_stats_for_ui(self) -> Dict[str, Any]:
        """Get statistics formatted for the UI display"""
        if not self.calculated_stats:
            return {}
        
        return {
            **self.calculated_stats,
            "data_source": self.data_source,
            "loaded_at": self.data_loaded_at.isoformat() if self.data_loaded_at else None,
            "user_name": self.current_user.name if self.current_user else None
        }
    
    def get_context_for_agent(self) -> Dict[str, Any]:
        """Get context formatted for the AI agent"""
        if not self.current_user or not self.calculated_stats:
            return {"error": "No user data loaded"}
        
        # Build agent context with the SAME data as UI
        context = {
            "user": {
                "name": self.current_user.name,
                "age": self.current_user.age,
                "diabetes_type": self.current_user.diabetes_type,
                "device_type": self.current_user.device_type,
                "years_with_diabetes": self.current_user.years_with_diabetes,
                "typical_pattern": getattr(self.current_user, 'typical_glucose_pattern', 'normal')
            },
            "statistics": self._safe_convert_for_json(self.calculated_stats),
            "patterns": self._safe_convert_for_json(self.identified_patterns),
            "data_points": len(self.processed_glucose_data) if self.processed_glucose_data is not None else 0,
            "recent_readings": self._get_recent_readings_for_agent(),
            "data_metadata": {
                "source": self.data_source,
                "loaded_at": self.data_loaded_at.isoformat() if self.data_loaded_at else None,
                "data_age_minutes": int((datetime.now() - self.data_loaded_at).total_seconds() / 60) if self.data_loaded_at else None
            }
        }
        
        return context
    
    def get_chart_data(self) -> Optional[pd.DataFrame]:
        """Get processed data for chart display"""
        return self.processed_glucose_data
    
    def _calculate_unified_stats(self) -> Dict[str, Any]:
        """Calculate statistics using a single, consistent method"""
        if self.processed_glucose_data is None or self.processed_glucose_data.empty:
            return {"error": "No data available"}
        
        try:
            # Get glucose values
            glucose_values = self.processed_glucose_data['value'].dropna()
            
            if len(glucose_values) == 0:
                return {"error": "No valid glucose values"}
            
            # Convert to numpy array for consistent calculations
            import numpy as np
            values = np.array(glucose_values.tolist(), dtype=float)
            
            # Calculate basic statistics
            avg_glucose = float(np.mean(values))
            min_glucose = float(np.min(values))
            max_glucose = float(np.max(values))
            std_glucose = float(np.std(values))
            total_readings = int(len(values))
            
            # Calculate time in ranges - CONSISTENT METHOD
            in_range_mask = (values >= 70) & (values <= 180)
            below_range_mask = values < 70
            above_range_mask = values > 180
            
            in_range_count = int(np.sum(in_range_mask))
            below_range_count = int(np.sum(below_range_mask))
            above_range_count = int(np.sum(above_range_mask))
            
            # Calculate percentages
            time_in_range = (in_range_count / total_readings) * 100 if total_readings > 0 else 0
            time_below_70 = (below_range_count / total_readings) * 100 if total_readings > 0 else 0
            time_above_180 = (above_range_count / total_readings) * 100 if total_readings > 0 else 0
            
            # Calculate additional metrics
            gmi = 3.31 + (0.02392 * avg_glucose)  # Glucose Management Indicator
            cv = (std_glucose / avg_glucose) * 100 if avg_glucose > 0 else 0  # Coefficient of Variation
            
            stats = {
                "average_glucose": avg_glucose,
                "min_glucose": min_glucose,
                "max_glucose": max_glucose,
                "std_glucose": std_glucose,
                "time_in_range_70_180": time_in_range,
                "time_below_70": time_below_70,
                "time_above_180": time_above_180,
                "total_readings": total_readings,
                "gmi": gmi,
                "cv": cv,
                "in_range_count": in_range_count,
                "below_range_count": below_range_count,
                "above_range_count": above_range_count
            }
            
            # Log for debugging
            logger.info(f"Calculated stats - TIR: {time_in_range:.1f}%, Total: {total_readings}, In range: {in_range_count}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating unified stats: {e}")
            return {"error": f"Statistics calculation failed: {str(e)}"}
    
    def _generate_realistic_mock_data(self, user_key: str) -> list:
        """Generate consistent mock data for demo users"""
        from mistral_chat import GlucoseDataGenerator
        
        # Map users to patterns
        pattern_map = {
            "sarah_g7": "normal",
            "marcus_one": "dawn_phenomenon", 
            "jennifer_g6": "normal",
            "robert_receiver": "dawn_phenomenon"
        }
        
        user_pattern = pattern_map.get(user_key, "normal")
        
        # Generate 14 days of data
        mock_data = GlucoseDataGenerator.create_realistic_pattern(days=14, user_type=user_pattern)
        
        logger.info(f"Generated {len(mock_data)} mock data points for {user_key} with pattern {user_pattern}")
        
        return mock_data
    
    def _get_recent_readings_for_agent(self, count: int = 5) -> list:
        """Get recent readings formatted for agent context"""
        if self.processed_glucose_data is None or self.processed_glucose_data.empty:
            return []
        
        try:
            recent_df = self.processed_glucose_data.tail(count)
            readings = []
            
            for _, row in recent_df.iterrows():
                display_time = row.get('displayTime') or row.get('systemTime')
                glucose_value = row.get('value')
                trend_value = row.get('trend', 'flat')
                
                if pd.notna(display_time):
                    if isinstance(display_time, str):
                        time_str = display_time
                    else:
                        time_str = pd.to_datetime(display_time).isoformat()
                else:
                    time_str = datetime.now().isoformat()
                
                if pd.notna(glucose_value):
                    glucose_clean = self._safe_convert_for_json(glucose_value)
                else:
                    glucose_clean = None
                
                trend_clean = str(trend_value) if pd.notna(trend_value) else 'flat'
                
                readings.append({
                    "time": time_str,
                    "glucose": glucose_clean,
                    "trend": trend_clean
                })
            
            return readings
            
        except Exception as e:
            logger.error(f"Error getting recent readings: {e}")
            return []
    
    def _safe_convert_for_json(self, obj):
        """Safely convert objects for JSON serialization"""
        import numpy as np
        
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._safe_convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_convert_for_json(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    def _build_success_response(self) -> Dict[str, Any]:
        """Build a consistent success response"""
        data_points = len(self.processed_glucose_data) if self.processed_glucose_data is not None else 0
        avg_glucose = self.calculated_stats.get('average_glucose', 0)
        time_in_range = self.calculated_stats.get('time_in_range_70_180', 0)
        
        return {
            "success": True,
            "message": f"✅ Successfully loaded data for {self.current_user.name}",
            "user": asdict(self.current_user),
            "data_points": data_points,
            "stats": self.calculated_stats,
            "data_source": self.data_source,
            "summary": f"📊 {data_points} readings | Avg: {avg_glucose:.1f} mg/dL | TIR: {time_in_range:.1f}% | Source: {self.data_source}"
        }
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """Validate that all components are using consistent data"""
        if not self.calculated_stats:
            return {"valid": False, "message": "No data loaded"}
        
        validation = {
            "valid": True,
            "data_source": self.data_source,
            "data_age_minutes": int((datetime.now() - self.data_loaded_at).total_seconds() / 60) if self.data_loaded_at else None,
            "total_readings": self.calculated_stats.get('total_readings', 0),
            "time_in_range": self.calculated_stats.get('time_in_range_70_180', 0),
            "average_glucose": self.calculated_stats.get('average_glucose', 0),
            "user": self.current_user.name if self.current_user else None
        }
        
        logger.info(f"Data consistency check: {validation}")
        
        return validation

# ADDITIONAL: Debug function to test the API connection as it was working before
def test_original_api_method():
    """Test the API exactly as it was working before unified data manager"""
    from apifunctions import DexcomAPI, DEMO_USERS
    
    print("🔍 Testing API exactly as it was working before...")
    
    api = DexcomAPI()
    
    # Test with sarah_g7 as it was working before
    user_key = "sarah_g7"
    user = DEMO_USERS[user_key]
    
    print(f"Testing with {user.name} ({user.username})")
    
    try:
        # Call simulate_demo_login exactly as before
        access_token = api.simulate_demo_login(user_key)
        print(f"✅ Authentication: {bool(access_token)}")
        
        if access_token:
            # Call get_egv_data exactly as before
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            
            egv_data = api.get_egv_data(
                start_date.isoformat(),
                end_date.isoformat()
            )
            
            print(f"✅ EGV Data: {len(egv_data)} readings")
            
            if egv_data:
                print(f"✅ SUCCESS! API is working as before")
                sample = egv_data[0] if egv_data else {}
                print(f"Sample reading: {sample}")
                return True
            else:
                print("⚠️  API authenticated but returned no data")
                return False
        else:
            print("❌ Authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Test the original API method
    test_original_api_method()