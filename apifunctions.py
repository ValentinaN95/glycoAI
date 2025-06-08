#!/usr/bin/env python3
"""
GlucoBuddy MCP Server
Implements Model Context Protocol server for glucose data analysis
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import asdict

# Import your existing classes
from apifunctions import DexcomAPI, GlucoseAnalyzer, DEMO_USERS, DemoUser

# MCP Protocol Types
class MCPRequest:
    def __init__(self, id: Union[str, int], method: str, params: Dict[str, Any] = None):
        self.id = id
        self.method = method
        self.params = params or {}

class MCPResponse:
    def __init__(self, id: Union[str, int], result: Any = None, error: Dict[str, Any] = None):
        self.id = id
        self.result = result
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        response = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response

class GlucoBuddyMCPServer:
    """MCP Server for GlucoBuddy glucose monitoring system"""
    
    def __init__(self):
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        self.current_user: Optional[DemoUser] = None
        self.current_glucose_data: Optional[pd.DataFrame] = None
        self.current_stats: Optional[Dict] = None
        self.current_patterns: Optional[Dict] = None
        
        # MCP Protocol setup
        self.tools = {
            "load_user_data": self._load_user_data,
            "get_glucose_summary": self._get_glucose_summary,
            "get_recent_readings": self._get_recent_readings,
            "analyze_patterns": self._analyze_patterns,
            "get_time_in_range": self._get_time_in_range,
            "get_user_info": self._get_user_info,
            "list_available_users": self._list_available_users
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        try:
            # Parse request
            req_id = request_data.get("id")
            method = request_data.get("method")
            params = request_data.get("params", {})
            
            self.logger.info(f"Handling request: {method}")
            
            if method == "initialize":
                return self._handle_initialize(req_id, params)
            elif method == "tools/list":
                return self._handle_tools_list(req_id)
            elif method == "tools/call":
                return await self._handle_tools_call(req_id, params)
            elif method == "resources/list":
                return self._handle_resources_list(req_id)
            elif method == "resources/read":
                return await self._handle_resources_read(req_id, params)
            else:
                return MCPResponse(req_id, error={
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }).to_dict()
                
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            return MCPResponse(request_data.get("id"), error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }).to_dict()

    def _handle_initialize(self, req_id: Union[str, int], params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return MCPResponse(req_id, result={
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {}
            },
            "serverInfo": {
                "name": "glucobuddy-mcp-server",
                "version": "1.0.0"
            }
        }).to_dict()

    def _handle_tools_list(self, req_id: Union[str, int]) -> Dict[str, Any]:
        """List available tools"""
        tools_list = [
            {
                "name": "load_user_data",
                "description": "Load glucose data for a specific demo user",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_key": {
                            "type": "string",
                            "description": "Demo user key (sarah_g7, marcus_one, jennifer_g6, robert_receiver)"
                        }
                    },
                    "required": ["user_key"]
                }
            },
            {
                "name": "get_glucose_summary",
                "description": "Get comprehensive glucose statistics summary",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_recent_readings",
                "description": "Get recent glucose readings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of recent readings to return",
                            "default": 10
                        }
                    }
                }
            },
            {
                "name": "analyze_patterns",
                "description": "Analyze glucose patterns and trends",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_time_in_range",
                "description": "Get detailed time in range analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "list_available_users",
                "description": "List all available demo users",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        return MCPResponse(req_id, result={"tools": tools_list}).to_dict()

    async def _handle_tools_call(self, req_id: Union[str, int], params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return MCPResponse(req_id, error={
                "code": -32602,
                "message": f"Unknown tool: {tool_name}"
            }).to_dict()
        
        try:
            result = await self.tools[tool_name](**arguments)
            return MCPResponse(req_id, result=[{
                "type": "text",
                "text": result
            }]).to_dict()
        except Exception as e:
            return MCPResponse(req_id, error={
                "code": -32603,
                "message": f"Tool execution failed: {str(e)}"
            }).to_dict()

    def _handle_resources_list(self, req_id: Union[str, int]) -> Dict[str, Any]:
        """List available resources"""
        resources = []
        
        if self.current_user:
            resources.extend([
                {
                    "uri": "glucose://current_user",
                    "name": f"Current User: {self.current_user.name}",
                    "description": "Currently loaded user information"
                },
                {
                    "uri": "glucose://glucose_data",
                    "name": "Current Glucose Data",
                    "description": "Loaded glucose readings and statistics"
                }
            ])
        
        return MCPResponse(req_id, result={"resources": resources}).to_dict()

    async def _handle_resources_read(self, req_id: Union[str, int], params: Dict[str, Any]) -> Dict[str, Any]:
        """Read resource content"""
        uri = params.get("uri")
        
        if uri == "glucose://current_user" and self.current_user:
            content = json.dumps(asdict(self.current_user), indent=2)
            return MCPResponse(req_id, result={
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content
                }]
            }).to_dict()
        
        elif uri == "glucose://glucose_data" and self.current_stats:
            content = json.dumps({
                "stats": self.current_stats,
                "patterns": self.current_patterns,
                "data_points": len(self.current_glucose_data) if self.current_glucose_data is not None else 0
            }, indent=2)
            return MCPResponse(req_id, result={
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json", 
                    "text": content
                }]
            }).to_dict()
        
        return MCPResponse(req_id, error={
            "code": -32602,
            "message": f"Resource not found: {uri}"
        }).to_dict()

    # Tool implementations
    async def _load_user_data(self, user_key: str) -> str:
        """Load glucose data for a specific demo user"""
        try:
            if user_key not in DEMO_USERS:
                return f"❌ Invalid user key. Available: {', '.join(DEMO_USERS.keys())}"
            
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
            
            return f"✅ Successfully loaded data for {self.current_user.name} ({self.current_user.device_type})\n" \
                   f"📊 Loaded {len(self.current_glucose_data)} glucose readings\n" \
                   f"📈 Average glucose: {self.current_stats.get('average_glucose', 0):.1f} mg/dL\n" \
                   f"🎯 Time in range: {self.current_stats.get('time_in_range_70_180', 0):.1f}%"
            
        except Exception as e:
            return f"❌ Failed to load user data: {str(e)}"

    async def _get_glucose_summary(self) -> str:
        """Get comprehensive glucose statistics"""
        if not self.current_user or not self.current_stats:
            return "❌ No user data loaded. Use load_user_data first."
        
        stats = self.current_stats
        
        return f"""📊 **Glucose Summary for {self.current_user.name}**

**Overall Metrics:**
• Average glucose: {stats.get('average_glucose', 0):.1f} mg/dL
• Glucose range: {stats.get('min_glucose', 0):.0f} - {stats.get('max_glucose', 0):.0f} mg/dL
• Standard deviation: {stats.get('std_glucose', 0):.1f} mg/dL
• Total readings: {stats.get('total_readings', 0)}

**Time in Range Analysis:**
• Time in range (70-180): {stats.get('time_in_range_70_180', 0):.1f}%
• Time below 70: {stats.get('time_below_70', 0):.1f}%  
• Time above 180: {stats.get('time_above_180', 0):.1f}%

**Assessment:**
{'✅ Excellent glucose control!' if stats.get('time_in_range_70_180', 0) > 80 else '⚠️ Good control with room for improvement' if stats.get('time_in_range_70_180', 0) > 60 else '🔴 Consider discussing with healthcare provider'}

*Data from {self.current_user.device_type}*"""

    async def _get_recent_readings(self, count: int = 10) -> str:
        """Get recent glucose readings"""
        if self.current_glucose_data is None or self.current_glucose_data.empty:
            return "❌ No glucose data available"
        
        recent_data = self.current_glucose_data.tail(count)
        
        result = f"📱 **Last {len(recent_data)} Glucose Readings**\n\n"
        
        for _, row in recent_data.iterrows():
            time_str = pd.to_datetime(row['displayTime']).strftime("%m/%d %H:%M")
            glucose = row['value']
            trend = row.get('trend', 'flat')
            
            trend_arrow = {
                'flat': '→', 'fortyFiveUp': '↗', 'singleUp': '↑', 'doubleUp': '⬆',
                'fortyFiveDown': '↘', 'singleDown': '↓', 'doubleDown': '⬇'
            }.get(trend, '→')
            
            result += f"• {time_str}: {glucose:.0f} mg/dL {trend_arrow}\n"
        
        return result

    async def _analyze_patterns(self) -> str:
        """Analyze glucose patterns"""
        if not self.current_patterns:
            return "❌ No pattern data available"
        
        patterns = self.current_patterns.get('patterns', [])
        
        if isinstance(patterns, list):
            pattern_text = "\n".join([f"• {pattern}" for pattern in patterns])
        else:
            pattern_text = str(patterns)
        
        return f"""🔍 **Glucose Pattern Analysis**

{pattern_text}

**Recommendations:**
• Monitor glucose trends around meal times
• Consider the impact of exercise and stress
• Review patterns with your healthcare provider
• Set up alerts for concerning trends

*Analysis based on {len(self.current_glucose_data) if self.current_glucose_data is not None else 0} data points*"""

    async def _get_time_in_range(self) -> str:
        """Get detailed time in range analysis"""
        if not self.current_stats:
            return "❌ No statistics available"
        
        stats = self.current_stats
        tir = stats.get('time_in_range_70_180', 0)
        low = stats.get('time_below_70', 0)
        high = stats.get('time_above_180', 0)
        
        return f"""🎯 **Time in Range Analysis**

**Current Performance:**
• Target range (70-180 mg/dL): {tir:.1f}%
• Below range (<70 mg/dL): {low:.1f}%
• Above range (>180 mg/dL): {high:.1f}%

**Clinical Targets:**
• Ideal TIR: >70%
• Hypoglycemia risk: <4% below 70
• Hyperglycemia: <25% above 180

**Your Status:**
{'🟢 Meeting clinical targets' if tir > 70 and low < 4 else '🟡 Close to targets' if tir > 60 else '🔴 Below recommended targets'}

**Recommendations:**
{'Continue current management strategy' if tir > 70 else 'Consider adjusting diabetes management plan with your healthcare provider'}"""

    async def _list_available_users(self) -> str:
        """List available demo users"""
        result = "👥 **Available Demo Users:**\n\n"
        
        for key, user in DEMO_USERS.items():
            result += f"**{key}**: {user.name}\n"
            result += f"  • Age: {user.age}, {user.diabetes_type}\n"
            result += f"  • Device: {user.device_type}\n"
            result += f"  • Experience: {user.years_with_diabetes} years with diabetes\n"
            result += f"  • Pattern: {user.typical_glucose_pattern}\n\n"
        
        return result

    async def _get_user_info(self) -> str:
        """Get current user information"""
        if not self.current_user:
            return "❌ No user loaded"
        
        user = self.current_user
        return f"""👤 **Current User Profile**

**Basic Information:**
• Name: {user.name}
• Age: {user.age}
• Diabetes Type: {user.diabetes_type}
• Years with diabetes: {user.years_with_diabetes}

**Device Information:**
• Device: {user.device_type}
• Typical pattern: {user.typical_glucose_pattern}

**Description:**
{user.description}"""

async def run_mcp_server():
    """Run the MCP server"""
    server = GlucoBuddyMCPServer()
    
    print("🩺 GlucoBuddy MCP Server starting...", file=sys.stderr)
    
    try:
        while True:
            # Read JSON-RPC request from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            
            if not line:
                break
                
            try:
                request = json.loads(line.strip())
                response = await server.handle_request(request)
                
                # Send response to stdout
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("🛑 MCP Server stopping...", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(run_mcp_server())