"""
GycoAI - AI-Powered Glucose Insights
Main Gradio application interface for demonstrating Dexcom API integration
and AI-powered glucose pattern analysis.
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Tuple

# Import our custom functions
from glucobuddy_functions import (
    DexcomAPI, 
    GlucoseAnalyzer,
    DEMO_USERS,
    generate_ai_insights,
    format_glucose_data_for_display
)


class GlucoBuddyApp:
    """Main application class for GlucoBuddy"""
    
    def __init__(self):
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        self.current_user = None
        self.glucose_data = None
        
    def select_demo_user(self, user_key: str) -> Tuple[str, str, str]:
        """Handle demo user selection and simulate login"""
        if user_key not in DEMO_USERS:
            return "❌ Invalid user selection", "", ""
        
        try:
            # Simulate getting access token for demo user
            user = DEMO_USERS[user_key]
            access_token = self.dexcom_api.simulate_demo_login(user_key)
            self.current_user = user
            
            # Get data range
            try:
                data_range = self.dexcom_api.get_data_range()
                range_info = f"Data available from {data_range.get('egvStart', 'Unknown')} to {data_range.get('egvEnd', 'Unknown')}"
            except:
                range_info = "Simulated data range: Last 30 days"
            
            success_msg = f"""
            ✅ **Successfully connected as {user.name}**
            
            **Device**: {user.device_type}
            **Description**: {user.description}
            **Status**: Connected to Dexcom Sandbox
            **Data Range**: {range_info}
            
            You can now load and analyze glucose data using the buttons below.
            """
            
            return success_msg, f"Connected: {user.name}", gr.update(visible=True)
            
        except Exception as e:
            return f"❌ Connection failed: {str(e)}", "", gr.update(visible=False)
    
    def load_glucose_data(self) -> Tuple[str, str]:
        """Load glucose data for the current user"""
        if not self.current_user:
            return "❌ Please select a demo user first", ""
        
        try:
            # Calculate date range (last 7 days for demo)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Get EGV data
            egv_data = self.dexcom_api.get_egv_data(
                start_date.isoformat(),
                end_date.isoformat()
            )
            
            if not egv_data:
                # Generate simulated data for demo
                egv_data = self._generate_demo_glucose_data()
            
            # Process the data
            self.glucose_data = self.analyzer.process_egv_data(egv_data)
            
            if self.glucose_data.empty:
                return "❌ No glucose data available", ""
            
            # Format data for display
            formatted_data = format_glucose_data_for_display(self.glucose_data)
            
            data_summary = f"""
            ## 📊 Data Loaded Successfully
            
            **Total Readings**: {len(self.glucose_data)}
            **Date Range**: {self.glucose_data['systemTime'].min().strftime('%Y-%m-%d')} to {self.glucose_data['systemTime'].max().strftime('%Y-%m-%d')}
            **Device**: {self.current_user.device_type}
            
            {formatted_data}
            """
            
            return data_summary, "✅ Data loaded - ready for analysis"
            
        except Exception as e:
            return f"❌ Failed to load glucose data: {str(e)}", ""
    
    def generate_insights(self) -> str:
        """Generate AI insights from glucose data"""
        if self.glucose_data is None or self.glucose_data.empty:
            return "❌ Please load glucose data first"
        
        if not self.current_user:
            return "❌ Please select a demo user first"
        
        try:
            # Calculate statistics
            stats = self.analyzer.calculate_basic_stats(self.glucose_data)
            patterns = self.analyzer.identify_patterns(self.glucose_data)
            
            # Generate AI insights
            insights = generate_ai_insights(stats, patterns, self.current_user)
            
            return insights
            
        except Exception as e:
            return f"❌ Failed to generate insights: {str(e)}"
    
    def create_glucose_chart(self) -> Optional[go.Figure]:
        """Create an interactive glucose chart"""
        if self.glucose_data is None or self.glucose_data.empty:
            return None
        
        fig = go.Figure()
        
        # Add glucose line
        fig.add_trace(go.Scatter(
            x=self.glucose_data['systemTime'],
            y=self.glucose_data['value'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4)
        ))
        
        # Add target range
        fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                     annotation_text="Target Range Lower (70 mg/dL)")
        fig.add_hline(y=180, line_dash="dash", line_color="orange",
                     annotation_text="Target Range Upper (180 mg/dL)")
        
        # Add critical levels
        fig.add_hline(y=54, line_dash="dot", line_color="red",
                     annotation_text="Severe Hypoglycemia (54 mg/dL)")
        fig.add_hline(y=250, line_dash="dot", line_color="red",
                     annotation_text="Severe Hyperglycemia (250 mg/dL)")
        
        # Styling
        fig.update_layout(
            title=f"Glucose Trends - {self.current_user.name if self.current_user else 'Demo User'}",
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _generate_demo_glucose_data(self) -> list:
        """Generate realistic demo glucose data"""
        import random
        import numpy as np
        
        demo_data = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(288 * 7):  # 5-minute intervals for 7 days
            timestamp = base_time + timedelta(minutes=i * 5)
            
            # Generate realistic glucose patterns
            hour = timestamp.hour
            
            # Base glucose with circadian rhythm
            base_glucose = 120 + 20 * np.sin((hour - 6) * np.pi / 12)
            
            # Add meal spikes
            if hour in [7, 12, 18]:  # Meal times
                base_glucose += random.randint(20, 60)
            
            # Add some random variation
            glucose = base_glucose + random.randint(-15, 15)
            glucose = max(50, min(300, glucose))  # Clamp to reasonable range
            
            # Simple trend calculation
            trend = "flat"
            if i > 0:
                prev_glucose = demo_data[-1]["value"]
                diff = glucose - prev_glucose
                if diff > 5:
                    trend = "singleUp"
                elif diff < -5:
                    trend = "singleDown"
            
            demo_data.append({
                "systemTime": timestamp.isoformat(),
                "displayTime": timestamp.isoformat(),
                "value": int(glucose),
                "trend": trend,
                "status": "ok"
            })
        
        return demo_data


def create_interface():
    """Create the Gradio interface"""
    app = GlucoBuddyApp()
    
    with gr.Blocks(title="GlycoAI - AI Glucose Insights", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("""
        # 🩺 GlycoAI
        ## AI-Powered Glucose Pattern Analysis
        
        Connect your Dexcom CGM data and get personalized insights powered by advanced AI analysis.
        """)
        
        # Demo User Selection Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👥 Select Demo User")
                gr.Markdown("Choose from our demo users to explore GlucoBuddy's capabilities:")
                
                with gr.Row():
                    sarah_btn = gr.Button("Sarah Thompson\n(G7 Mobile)", variant="secondary")
                    marcus_btn = gr.Button("Marcus Rodriguez\n(ONE+ Mobile)", variant="secondary")
                    jennifer_btn = gr.Button("Jennifer Chen\n(G6 Mobile)", variant="secondary")
                    robert_btn = gr.Button("Robert Williams\n(G6 Receiver)", variant="secondary")
        
        # Connection Status
        connection_status = gr.Textbox(
            label="Connection Status",
            value="No user selected",
            interactive=False
        )
        
        # Main Interface (initially hidden)
        with gr.Group(visible=False) as main_interface:
            
            # User Info and Data Loading
            with gr.Row():
                with gr.Column():
                    user_info = gr.Markdown("", label="User Information")
                    
                with gr.Column():
                    load_data_btn = gr.Button("📊 Load Glucose Data", variant="primary", size="lg")
                    data_status = gr.Textbox(
                        label="Data Status",
                        value="Ready to load data",
                        interactive=False
                    )
            
            # Data Display and Analysis
            with gr.Tabs():
                
                with gr.TabItem("📈 Glucose Chart"):
                    glucose_chart = gr.Plot(label="Glucose Trends")
                    refresh_chart_btn = gr.Button("🔄 Refresh Chart")
                
                with gr.TabItem("📋 Data Overview"):
                    data_display = gr.Markdown("", label="Glucose Data")
                
                with gr.TabItem("🤖 AI Insights"):
                    with gr.Row():
                        generate_insights_btn = gr.Button("🧠 Generate AI Insights", variant="primary", size="lg")
                    
                    ai_insights = gr.Markdown("", label="AI Analysis")
        
        # Event Handlers
        def handle_user_selection(user_key):
            status, conn_status, interface_visibility = app.select_demo_user(user_key)
            return status, conn_status, interface_visibility
        
        # User selection buttons
        sarah_btn.click(
            lambda: handle_user_selection("sarah_g7"),
            outputs=[user_info, connection_status, main_interface]
        )
        
        marcus_btn.click(
            lambda: handle_user_selection("marcus_one"),
            outputs=[user_info, connection_status, main_interface]
        )
        
        jennifer_btn.click(
            lambda: handle_user_selection("jennifer_g6"),
            outputs=[user_info, connection_status, main_interface]
        )
        
        robert_btn.click(
            lambda: handle_user_selection("robert_receiver"),
            outputs=[user_info, connection_status, main_interface]
        )
        
        # Data loading
        load_data_btn.click(
            app.load_glucose_data,
            outputs=[data_display, data_status]
        )
        
        # Chart generation
        refresh_chart_btn.click(
            app.create_glucose_chart,
            outputs=[glucose_chart]
        )
        
        # AI insights generation
        generate_insights_btn.click(
            app.generate_insights,
            outputs=[ai_insights]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Note**: This is a demonstration using Dexcom Sandbox data. 
        For real clinical use, always consult with your healthcare provider.
        
        GlycoAI - Empowering diabetes management through AI insights 🌟
        """)
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        debug=True
    )