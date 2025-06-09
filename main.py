"""
GycoAI - AI-Powered Glucose Insights
Main Gradio application interface for demonstrating Dexcom API integration
and AI-powered glucose pattern analysis with MistralMCPClient.
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Tuple, List
import asyncio
import requests
import logging

# Import the Mistral chat class
from mistral_chat import GlucoBuddyMistralChat

# Initialize Mistral chat with your agent ID
MISTRAL_API_KEY = "ZAjtPftvZrCxK7WWwjBJIYudaiNhwRuO"
MISTRAL_AGENT_ID = "ag:2d7a33b1:20250608:glycoaiagent:cc72ded9"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom functions
from apifunctions import (
    DexcomAPI,
    GlucoseAnalyzer,
    DEMO_USERS,
    format_glucose_data_for_display
)

class GlucoBuddyApp:
    """Main application class for GlucoBuddy"""

    def __init__(self):
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        self.current_user = None
        self.glucose_data = None
        self.mistral_chat = GlucoBuddyMistralChat(MISTRAL_API_KEY, MISTRAL_AGENT_ID)
        self.chat_history = []

    def select_demo_user(self, user_key: str) -> Tuple[str, str, str]:
        """Handle demo user selection and simulate login"""
        if user_key not in DEMO_USERS:
            return "❌ Invalid user selection", "", gr.update(visible=False)

        try:
            user = DEMO_USERS[user_key]
            access_token = self.dexcom_api.simulate_demo_login(user_key)
            self.current_user = user

            # Load user data in Mistral chat
            load_result = self.mistral_chat.load_user_data(user_key)
            
            if not load_result['success']:
                logger.warning(f"Failed to load user data for Mistral: {load_result['message']}")

            # Clear chat history when switching users
            self.chat_history = []
            self.mistral_chat.clear_conversation()

            return (
                f"Connected: {user.name} ({user.device_type})", 
                f"Ready to chat with {user.name}", 
                gr.update(visible=True)
            )

        except Exception as e:
            logger.error(f"User selection failed: {str(e)}")
            return f"❌ Connection failed: {str(e)}", "", gr.update(visible=False)

    def load_glucose_data(self) -> Tuple[str, str, go.Figure]:
        """Load glucose data for the current user"""
        if not self.current_user:
            return "Please select a demo user first", "", None

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)

            egv_data = self.dexcom_api.get_egv_data(
                start_date.isoformat(),
                end_date.isoformat()
            )

            if not egv_data:
                egv_data = self._generate_demo_glucose_data()

            self.glucose_data = self.analyzer.process_egv_data(egv_data)

            if self.glucose_data.empty:
                return "No glucose data available", "", None

            # Calculate statistics
            avg_glucose = self.glucose_data['value'].mean()
            std_glucose = self.glucose_data['value'].std()
            min_glucose = self.glucose_data['value'].min()
            max_glucose = self.glucose_data['value'].max()
            readings_count = len(self.glucose_data)

            in_range = ((self.glucose_data['value'] >= 70) & (self.glucose_data['value'] <= 180)).sum()
            below_range = (self.glucose_data['value'] < 70).sum()
            above_range = (self.glucose_data['value'] > 180).sum()

            time_in_range = (in_range / readings_count) * 100
            time_below_range = (below_range / readings_count) * 100
            time_above_range = (above_range / readings_count) * 100

            gmi = 3.31 + (0.02392 * avg_glucose)
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

            chart = self.create_glucose_chart()
            return data_summary, "✅ Data loaded - ready for chat", chart

        except Exception as e:
            logger.error(f"Failed to load glucose data: {str(e)}")
            return f"Failed to load glucose data: {str(e)}", "", None

    def get_template_prompts(self) -> List[str]:
        """Get template prompts based on user data"""
        if not self.current_user:
            return [
                "What should I know about managing my diabetes?",
                "How can I improve my glucose control?"
            ]

        # Get current stats from Mistral chat if available
        if hasattr(self.mistral_chat, 'current_stats') and self.mistral_chat.current_stats:
            stats = self.mistral_chat.current_stats
            time_in_range = stats.get('time_in_range_70_180', 0)
            time_below_70 = stats.get('time_below_70', 0)

            templates = []

            if time_in_range < 70:
                templates.append("My time in range is below target. What specific strategies can help me improve it?")
            else:
                templates.append("My time in range looks good. How can I maintain this level of control?")

            if time_below_70 > 4:
                templates.append("I'm experiencing frequent low glucose episodes. What can I do to prevent them?")
            else:
                templates.append("What are the best practices for preventing hypoglycemia in my situation?")

            return templates
        
        return [
            "Can you analyze my recent glucose patterns and give me insights?",
            "What can I do to improve my diabetes management based on my data?"
        ]

    def chat_with_mistral(self, message: str, history: List) -> Tuple[str, List]:
        """Handle chat interaction with Mistral"""
        if not message.strip():
            return "", history

        if not self.current_user:
            response = "Please select a demo user first to get personalized insights about glucose data."
            history.append([message, response])
            return "", history

        try:
            # Send message to Mistral chat
            result = self.mistral_chat.chat_with_mistral(message)
            
            if result['success']:
                response = result['response']
                
                # Add context note if no user data was included
                if not result.get('context_included', True):
                    response += "\n\n💡 *For more personalized advice, make sure your glucose data is loaded.*"
            else:
                response = f"I apologize, but I encountered an error: {result.get('error', 'Unknown error')}. Please try again or rephrase your question."

            history.append([message, response])
            return "", history

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            error_response = f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."
            history.append([message, error_response])
            return "", history

    def use_template_prompt(self, template_text: str) -> str:
        """Use a template prompt in the chat"""
        return template_text

    def clear_chat_history(self) -> List:
        """Clear chat history"""
        self.chat_history = []
        self.mistral_chat.clear_conversation()
        return []

    def create_glucose_chart(self) -> Optional[go.Figure]:
        """Create an interactive glucose chart"""
        if self.glucose_data is None or self.glucose_data.empty:
            return None

        fig = go.Figure()

        # Color code based on glucose ranges
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
            hour = timestamp.hour

            # Create realistic glucose patterns
            base_glucose = 120 + 20 * np.sin((hour - 6) * np.pi / 12)

            # Add meal spikes
            if hour in [7, 12, 18]:
                base_glucose += random.randint(20, 60)

            glucose = base_glucose + random.randint(-15, 15)
            glucose = max(50, min(300, glucose))

            # Determine trend
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
    .template-button {
        margin: 0.25rem;
        padding: 0.5rem;
        font-size: 0.9rem;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    """

    with gr.Blocks(
        title="GlycoAI - AI Glucose Insights",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:

        # Header
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
                            <p style="margin: 0; font-size: 1.2rem; color: white; opacity: 0.9;">AI-Powered Glucose Chatbot</p>
                        </div>
                    </div>
                    <p style="margin-top: 1rem; font-size: 1rem; color: white; opacity: 0.8;">
                        Connect your Dexcom CGM data and chat with AI for personalized glucose insights
                    </p>
                </div>
                """)

        # User Selection Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👥 Select Demo User")
                gr.Markdown("Choose from our demo users to explore GlycoAI's chat capabilities")

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

        # Connection Status
        with gr.Row():
            connection_status = gr.Textbox(
                label="Current User",
                value="No user selected",
                interactive=False,
                container=True
            )

        # Main Interface (hidden until user is selected)
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

            # Main Content Tabs
            with gr.Tabs():

                # Glucose Chart Tab
                with gr.TabItem("📈 Glucose Chart"):
                    glucose_chart = gr.Plot(
                        label="Interactive Glucose Trends",
                        container=True
                    )

                # Chat Tab
                with gr.TabItem("💬 Chat with AI"):
                    with gr.Column(elem_classes=["chat-container"]):
                        gr.Markdown("### 🤖 Chat with GlycoAI about your glucose data")
                        
                        # Template Prompts
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**💡 Quick Start Templates:**")
                                with gr.Row():
                                    template1_btn = gr.Button(
                                        "🎯 Analyze My Patterns",
                                        variant="secondary",
                                        size="sm",
                                        elem_classes=["template-button"]
                                    )
                                    template2_btn = gr.Button(
                                        "⚡ Improve My Control",
                                        variant="secondary",
                                        size="sm",
                                        elem_classes=["template-button"]
                                    )
                                    template3_btn = gr.Button(
                                        "🍽️ Meal Management Tips",
                                        variant="secondary",
                                        size="sm",
                                        elem_classes=["template-button"]
                                    )

                        # Chat Interface
                        chatbot = gr.Chatbot(
                            label="💬 Chat with GlycoAI",
                            height=500,
                            show_label=True,
                            container=True,
                            bubble_full_width=False,
                            avatar_images=(None, "🩺")
                        )

                        # Chat Input
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Ask me about your glucose patterns, trends, or management strategies...",
                                label="Your Question",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button(
                                "Send 💬",
                                variant="primary",
                                scale=1
                            )

                        # Chat Controls
                        with gr.Row():
                            clear_chat_btn = gr.Button(
                                "🗑️ Clear Chat",
                                variant="secondary",
                                size="sm"
                            )
                            gr.Markdown("*AI responses are for informational purposes only. Always consult your healthcare provider.*")

                # Data Overview Tab
                with gr.TabItem("📋 Data Overview"):
                    data_display = gr.Markdown("Load data to see overview", container=True)

        # Event Handlers
        def handle_user_selection(user_key):
            status, data_status, interface_visibility = app.select_demo_user(user_key)
            return status, data_status, interface_visibility, []

        def get_template_prompt(template_type):
            templates = app.get_template_prompts()
            if template_type == 1:
                return templates[0] if templates else "Can you analyze my recent glucose patterns and give me insights?"
            elif template_type == 2:
                return templates[1] if len(templates) > 1 else "What can I do to improve my diabetes management based on my data?"
            else:
                return "What are some meal management strategies for better glucose control?"

        def handle_chat_submit(message, history):
            return app.chat_with_mistral(message, history)

        def handle_enter_key(message, history):
            if message.strip():
                return app.chat_with_mistral(message, history)
            return "", history

        # Connect Event Handlers
        user_selection_outputs = [connection_status, data_status, main_interface, chatbot]

        sarah_btn.click(
            lambda: handle_user_selection("sarah_g7"),
            outputs=user_selection_outputs
        )

        marcus_btn.click(
            lambda: handle_user_selection("marcus_one"),
            outputs=user_selection_outputs
        )

        jennifer_btn.click(
            lambda: handle_user_selection("jennifer_g6"),
            outputs=user_selection_outputs
        )

        robert_btn.click(
            lambda: handle_user_selection("robert_receiver"),
            outputs=user_selection_outputs
        )

        # Data Loading
        load_data_btn.click(
            app.load_glucose_data,
            outputs=[data_display, data_status, glucose_chart]
        )

        # Chat Handlers
        send_btn.click(
            handle_chat_submit,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )

        chat_input.submit(
            handle_enter_key,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )

        # Template Button Handlers
        template1_btn.click(
            lambda: get_template_prompt(1),
            outputs=[chat_input]
        )

        template2_btn.click(
            lambda: get_template_prompt(2),
            outputs=[chat_input]
        )

        template3_btn.click(
            lambda: get_template_prompt(3),
            outputs=[chat_input]
        )

        # Clear Chat
        clear_chat_btn.click(
            app.clear_chat_history,
            outputs=[chatbot]
        )

        # Footer
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; padding: 2rem; margin-top: 2rem; border-top: 1px solid #dee2e6; color: #6c757d;">
                <p><strong>⚠️ Important Medical Disclaimer</strong></p>
                <p>GlycoAI is for informational and educational purposes only. Always consult your healthcare provider 
                before making any changes to your diabetes management plan. This tool does not replace professional medical advice.</p>
                <p style="margin-top: 1rem; font-size: 0.9rem;">
                    🔒 Your data is processed securely and not stored permanently. 
                    💡 Powered by Dexcom API integration and Mistral AI.
                </p>
            </div>
            """)

    return interface


def main():
    """Main function to launch the application"""
    print("🚀 Starting GlycoAI - AI-Powered Glucose Insights...")
    
    try:
        # Create and launch the interface
        demo = create_interface()
        
        # Launch with custom settings
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default Gradio port
            share=True,            # Set to True for public sharing
            debug=True,             # Enable debug mode
            show_error=True,        # Show errors in the interface
            auth=None,              # No authentication required
            favicon_path=None,      # Use default favicon
            ssl_verify=False        # Disable SSL verification for development
        )
        
    except Exception as e:
        logger.error(f"Failed to launch GlycoAI application: {e}")
        print(f"❌ Error launching application: {e}")
        raise


if __name__ == "__main__":
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('glycoai.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run the main application
    main()