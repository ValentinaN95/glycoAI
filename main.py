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
from apifunctions import (
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
            return gr.Info("❌ Invalid user selection"), "", ""
        
        try:
            # Simulate getting access token for demo user
            user = DEMO_USERS[user_key]
            access_token = self.dexcom_api.simulate_demo_login(user_key)
            self.current_user = user
            
            # Show success notification
            gr.Info(f"✅ Successfully connected as {user.name}")
            
            return f"Connected: {user.name} ({user.device_type})", f"Ready to load data for {user.name}", gr.update(visible=True)
            
        except Exception as e:
            gr.Error(f"❌ Connection failed: {str(e)}")
            return "", "", gr.update(visible=False)
    
    def load_glucose_data(self) -> Tuple[str, str, go.Figure]:
        """Load glucose data for the current user"""
        if not self.current_user:
            gr.Warning("Please select a demo user first")
            return "Please select a demo user first", "", None
        
        try:
            # Calculate date range (last 14 days for demo)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            
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
                gr.Warning("No glucose data available")
                return "No glucose data available", "", None
            
            # Calculate comprehensive statistics
            avg_glucose = self.glucose_data['value'].mean()
            std_glucose = self.glucose_data['value'].std()
            min_glucose = self.glucose_data['value'].min()
            max_glucose = self.glucose_data['value'].max()
            readings_count = len(self.glucose_data)
            
            # Calculate time in range percentages
            in_range = ((self.glucose_data['value'] >= 70) & (self.glucose_data['value'] <= 180)).sum()
            below_range = (self.glucose_data['value'] < 70).sum()
            above_range = (self.glucose_data['value'] > 180).sum()
            
            time_in_range = (in_range / readings_count) * 100
            time_below_range = (below_range / readings_count) * 100
            time_above_range = (above_range / readings_count) * 100
            
            # Calculate GMI (Glucose Management Indicator)
            # GMI formula: 3.31 + (0.02392 × average_glucose_mg_dL)
            gmi = 3.31 + (0.02392 * avg_glucose)
            
            # Calculate coefficient of variation
            cv = (std_glucose / avg_glucose) * 100
            
            data_summary = f"""
## 📊 Data Summary for {self.current_user.name}

### Basic Information
• **Analysis Period:** {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')} (14 days)
• **Total Readings:** {readings_count:,} glucose measurements
• **Device:** {self.current_user.device_type}

### Glucose Statistics
• **Average Glucose:** {avg_glucose:.1f} mg/dL
• **Standard Deviation:** {std_glucose:.1f} mg/dL
• **Coefficient of Variation:** {cv:.1f}%
• **Glucose Range:** {min_glucose} - {max_glucose} mg/dL
• **GMI (Glucose Management Indicator):** {gmi:.1f}%

### Time in Range Analysis
• **Time in Range (70-180 mg/dL):** {time_in_range:.1f}%
• **Time Below Range (<70 mg/dL):** {time_below_range:.1f}%
• **Time Above Range (>180 mg/dL):** {time_above_range:.1f}%

### Clinical Targets
• **Target Time in Range:** >70% (Current: {time_in_range:.1f}%)
• **Target Time Below Range:** <4% (Current: {time_below_range:.1f}%)
• **Target CV:** <36% (Current: {cv:.1f}%)
            """
            
            # Create the chart automatically
            chart = self.create_glucose_chart()
            
            gr.Info("✅ Data loaded successfully!")
            return data_summary, "✅ Data loaded - ready for analysis", chart
            
        except Exception as e:
            gr.Error(f"Failed to load glucose data: {str(e)}")
            return f"Failed to load glucose data: {str(e)}", "", None
    
    def generate_insights(self) -> str:
        """Generate AI insights from glucose data"""
        if self.glucose_data is None or self.glucose_data.empty:
            gr.Warning("Please load glucose data first")
            return "Please load glucose data first"
        
        if not self.current_user:
            gr.Warning("Please select a demo user first")
            return "Please select a demo user first"
        
        try:
            # Calculate statistics
            stats = self.analyzer.calculate_basic_stats(self.glucose_data)
            patterns = self.analyzer.identify_patterns(self.glucose_data)
            
            # Generate AI insights
            insights = generate_ai_insights(stats, patterns, self.current_user)
            
            gr.Info("✅ AI insights generated!")
            return insights
            
        except Exception as e:
            gr.Error(f"Failed to generate insights: {str(e)}")
            return f"Failed to generate insights: {str(e)}"
    
    def create_glucose_chart(self) -> Optional[go.Figure]:
        """Create an interactive glucose chart"""
        if self.glucose_data is None or self.glucose_data.empty:
            return None
        
        fig = go.Figure()
        
        # Add glucose line with color coding
        colors = []
        for value in self.glucose_data['value']:
            if value < 70:
                colors.append('#E74C3C')  # Red for low
            elif value > 180:
                colors.append('#F39C12')  # Orange for high
            else:
                colors.append('#27AE60')  # Green for in range
        
        fig.add_trace(go.Scatter(
            x=self.glucose_data['systemTime'],
            y=self.glucose_data['value'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4, color=colors),
            hovertemplate='<b>%{y} mg/dL</b><br>%{x}<extra></extra>'
        ))
        
        # Add target range shading
        fig.add_hrect(
            y0=70, y1=180,
            fillcolor="rgba(39, 174, 96, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="Target Range",
            annotation_position="top left"
        )
        
        # Add reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="#E67E22", 
                     annotation_text="Low (70 mg/dL)", annotation_position="right")
        fig.add_hline(y=180, line_dash="dash", line_color="#E67E22",
                     annotation_text="High (180 mg/dL)", annotation_position="right")
        fig.add_hline(y=54, line_dash="dot", line_color="#E74C3C",
                     annotation_text="Severe Low (54 mg/dL)", annotation_position="right")
        fig.add_hline(y=250, line_dash="dot", line_color="#E74C3C",
                     annotation_text="Severe High (250 mg/dL)", annotation_position="right")
        
        # Styling
        fig.update_layout(
            title={
                'text': f"Glucose Trends - {self.current_user.name if self.current_user else 'Demo User'}",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            hovermode='x unified',
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def _generate_demo_glucose_data(self) -> list:
        """Generate realistic demo glucose data"""
        import random
        import numpy as np
        
        demo_data = []
        base_time = datetime.now() - timedelta(days=14)
        
        for i in range(288 * 14):  # 5-minute intervals for 14 days
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
    
    # Custom CSS for better styling
    custom_css = """
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .user-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(
        title="GlycoAI - AI Glucose Insights", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        # Header with logo placeholder
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="main-header">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 1rem;">
                        <div style="width: 60px; height: 60px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                            <span style="color: #667eea; font-size: 24px; font-weight: bold;">🩺</span>
                        </div>
                        <div>
                            <h1 style="margin: 0; font-size: 2.5rem; color: white;">GlycoAI</h1>
                            <p style="margin: 0; font-size: 1.2rem; color: white; opacity: 0.9;">AI-Powered Glucose Pattern Analysis</p>
                        </div>
                    </div>
                    <p style="margin-top: 1rem; font-size: 1rem; color: white; opacity: 0.8;">
                        Connect your Dexcom CGM data and get personalized insights powered by advanced AI analysis
                    </p>
                </div>
                """)
        
        # Demo User Selection Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👥 Select Demo User")
                gr.Markdown("Choose from our demo users to explore GlycoAI's capabilities")
                
                with gr.Row():
                    sarah_btn = gr.Button(
                        "Sarah Thompson\n(G7 Mobile)", 
                        variant="secondary",
                        size="lg"
                    )
                    marcus_btn = gr.Button(
                        "Marcus Rodriguez\n(ONE+ Mobile)", 
                        variant="secondary",
                        size="lg"
                    )
                    jennifer_btn = gr.Button(
                        "Jennifer Chen\n(G6 Mobile)", 
                        variant="secondary",
                        size="lg"
                    )
                    robert_btn = gr.Button(
                        "Robert Williams\n(G6 Receiver)", 
                        variant="secondary",
                        size="lg"
                    )
        
        # Connection Status - more compact
        with gr.Row():
            connection_status = gr.Textbox(
                label="Current User",
                value="No user selected",
                interactive=False,
                container=True
            )
        
        # Main Interface (initially hidden)
        with gr.Group(visible=False) as main_interface:
            
            # Data Loading Section
            with gr.Row():
                with gr.Column(scale=3):
                    data_status = gr.Textbox(
                        label="Data Status",
                        value="Ready to load data",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    load_data_btn = gr.Button(
                        "📊 Load Glucose Data\n(Last 14 Days)", 
                        variant="primary", 
                        size="lg"
                    )
            
            # Data Display and Analysis
            with gr.Tabs():
                
                with gr.TabItem("📈 Glucose Chart"):
                    glucose_chart = gr.Plot(
                        label="Interactive Glucose Trends",
                        container=True
                    )
                
                with gr.TabItem("🤖 AI Insights"):
                    with gr.Row():
                        generate_insights_btn = gr.Button(
                            "🧠 Generate AI Insights", 
                            variant="primary", 
                            size="lg"
                        )
                    
                    ai_insights = gr.Markdown("Load data and generate insights to see AI analysis", container=True)
                
                with gr.TabItem("📋 Data Overview"):
                    data_display = gr.Markdown("Load data to see overview", container=True)
        
        # Event Handlers
        def handle_user_selection(user_key):
            status, data_status, interface_visibility = app.select_demo_user(user_key)
            return status, data_status, interface_visibility
        
        # User selection buttons
        sarah_btn.click(
            lambda: handle_user_selection("sarah_g7"),
            outputs=[connection_status, data_status, main_interface]
        )
        
        marcus_btn.click(
            lambda: handle_user_selection("marcus_one"),
            outputs=[connection_status, data_status, main_interface]
        )
        
        jennifer_btn.click(
            lambda: handle_user_selection("jennifer_g6"),
            outputs=[connection_status, data_status, main_interface]
        )
        
        robert_btn.click(
            lambda: handle_user_selection("robert_receiver"),
            outputs=[connection_status, data_status, main_interface]
        )
        
        # Data loading - now also updates chart automatically
        load_data_btn.click(
            app.load_glucose_data,
            outputs=[data_display, data_status, glucose_chart]
        )
        
        # AI insights generation
        generate_insights_btn.click(
            app.generate_insights,
            outputs=[ai_insights]
        )
        
        # Footer
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
                <p style="margin: 0; color: #6c757d;">
                    <strong>Note:</strong> This is a demonstration using Dexcom Sandbox data.<br>
                    For real clinical use, always consult with your healthcare provider.
                </p>
                <p style="margin: 1rem 0 0 0; color: #495057; font-weight: bold;">
                    GlycoAI - Empowering diabetes management through AI insights ✨
                </p>
            </div>
            """)
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        debug=True
    )