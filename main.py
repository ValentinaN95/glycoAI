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

# Import our custom functions
from apifunctions import (
    DexcomAPI,
    GlucoseAnalyzer,
    DEMO_USERS,
    format_glucose_data_for_display
)

# Setup logging
logger = logging.getLogger(__name__)

class MistralMCPClient:
    """Mistral MCP Client for glucose data chatbot"""

    def __init__(self, api_key: str, agent_id: str = "ag:2d7a33b1:20250608:untitled-agent:4d4b9dcf"):
        self.api_key = api_key
        self.agent_id = agent_id
        self.session_id = None
        self.conversation_history = []
        self.base_url = "https://api.mistral.ai"
        
        # Initialize Dexcom API and analyzer
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        self.current_user = None
        self.current_glucose_data = None
        self.current_stats = None
        self.current_patterns = None

    async def initialize_session(self):
        """Initialize a new MCP session"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "agent_id": self.agent_id
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/agents/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            self.session_id = result.get("id")
            logger.info(f"Initialized MCP session: {self.session_id}")
            return self.session_id
        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            raise

    async def load_user_data(self, demo_user_key: str):
        """Load Dexcom data for a specific demo user"""
        try:
            if demo_user_key not in DEMO_USERS:
                raise ValueError(f"Invalid demo user: {demo_user_key}")
            
            self.current_user = DEMO_USERS[demo_user_key]
            
            # Authenticate with Dexcom API
            logger.info(f"Loading data for {self.current_user.name}...")
            self.dexcom_api.simulate_demo_login(demo_user_key)
            
            # Get glucose data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            
            egv_data = self.dexcom_api.get_egv_data(
                start_date.isoformat(),
                end_date.isoformat()
            )
            
            if not egv_data:
                egv_data = self._generate_demo_glucose_data()
            
            # Process the data
            self.current_glucose_data = self.analyzer.process_egv_data(egv_data)
            self.current_stats = self.analyzer.calculate_basic_stats(self.current_glucose_data)
            self.current_patterns = self.analyzer.identify_patterns(self.current_glucose_data)
            
            logger.info(f"Successfully loaded data for {self.current_user.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            return False

    def _generate_demo_glucose_data(self) -> list:
        """Generate realistic demo glucose data"""
        import random
        import numpy as np

        demo_data = []
        base_time = datetime.now() - timedelta(days=14)

        for i in range(288 * 14):
            timestamp = base_time + timedelta(minutes=i * 5)
            hour = timestamp.hour

            base_glucose = 120 + 20 * np.sin((hour - 6) * np.pi / 12)

            if hour in [7, 12, 18]:
                base_glucose += random.randint(20, 60)

            glucose = base_glucose + random.randint(-15, 15)
            glucose = max(50, min(300, glucose))

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

    def _prepare_glucose_context(self) -> str:
        """Prepare glucose data context for the AI"""
        if not self.current_user or not self.current_stats:
            return "No glucose data available."
        
        context = f"""
        Current patient: {self.current_user.name}
        Device: {self.current_user.device_type}
        Diabetes type: {getattr(self.current_user, 'diabetes_type', 'Not specified')}
        Years with diabetes: {getattr(self.current_user, 'years_with_diabetes', 'Not specified')}
        
        Recent glucose statistics:
        - Average glucose: {self.current_stats.get('average_glucose', 0):.1f} mg/dL
        - Time in range (70-180): {self.current_stats.get('time_in_range_70_180', 0):.1f}%
        - Time below 70: {self.current_stats.get('time_below_70', 0):.1f}%
        - Time above 180: {self.current_stats.get('time_above_180', 0):.1f}%
        - Glucose variability (std): {self.current_stats.get('std_glucose', 0):.1f} mg/dL
        
        Identified patterns:
        {self._format_patterns_for_context()}
        """
        
        return context

    def _format_patterns_for_context(self) -> str:
        """Format patterns for AI context"""
        if not self.current_patterns:
            return "No patterns identified"
        
        patterns = self.current_patterns.get('patterns', [])
        if isinstance(patterns, list):
            return "\n".join([f"- {pattern}" for pattern in patterns])
        else:
            return str(patterns)

    async def send_message(self, user_message: str) -> str:
        """Send a message to the Mistral MCP agent"""
        if not self.session_id:
            await self.initialize_session()
        
        # Prepare the system context with glucose data
        glucose_context = self._prepare_glucose_context()
        
        system_message = f"""You are GlycoAI, an AI assistant specialized in diabetes management and glucose monitoring. 
        You have access to the user's CGM (Continuous Glucose Monitor) data and can provide personalized insights.

        IMPORTANT: Always remind users that this is for informational purposes only and they should consult their healthcare provider for medical decisions.

        Current glucose data context:
        {glucose_context}

        Provide helpful, empathetic, and actionable responses about glucose management, patterns, and diabetes care.
        Use emojis to make responses more engaging and friendly.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Prepare messages for the API
        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history[-10:]  # Keep last 10 messages for context
        ]

        payload = {
            "agent_id": self.agent_id,
            "messages": messages,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/agents/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            ai_response = result["choices"][0]["message"]["content"]
            
            # Add AI response to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Failed to send message to MCP agent: {e}")
            return self._fallback_response(user_message)

    def _fallback_response(self, user_message: str) -> str:
        """Provide a fallback response when MCP is unavailable"""
        if not self.current_user:
            return "🤖 Hi! I'm GlycoAI. Please load your glucose data first so I can help you with personalized insights."
        
        # Simple keyword-based responses
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return f"👋 Hello {self.current_user.name}! I'm here to help you understand your glucose data. What would you like to know about your recent readings?"
        
        elif any(word in message_lower for word in ["summary", "overview", "status"]):
            if self.current_stats:
                tir = self.current_stats.get('time_in_range_70_180', 0)
                avg_glucose = self.current_stats.get('average_glucose', 0)
                return f"""📊 **Glucose Summary for {self.current_user.name}**
                
• Average glucose: {avg_glucose:.1f} mg/dL
• Time in range: {tir:.1f}%
• {'✅ Great control!' if tir > 70 else '⚠️ Room for improvement' if tir > 50 else '🔴 Consider discussing with your doctor'}

Remember to consult your healthcare provider about any concerns! 🩺"""
        
        elif any(word in message_lower for word in ["low", "hypoglycemia", "hypo"]):
            if self.current_stats:
                time_below = self.current_stats.get('time_below_70', 0)
                return f"""🩺 **Hypoglycemia Analysis**
                
You spent {time_below:.1f}% of time below 70 mg/dL.
{'✅ Low risk - good job!' if time_below < 4 else '⚠️ Monitor closely and consider discussing patterns with your healthcare provider' if time_below < 10 else '🚨 Higher risk detected - please consult your healthcare provider'}

Always consult your doctor about low glucose episodes! 💊"""
        
        else:
            return f"🤖 I understand you're asking about: '{user_message}'. While I have your glucose data loaded, I'm currently in basic mode. I can provide information about your glucose summary, patterns, or specific concerns. What would you like to know?"


class GlucoBuddyApp:
    """Main application class for GlucoBuddy"""

    def __init__(self):
        self.dexcom_api = DexcomAPI()
        self.analyzer = GlucoseAnalyzer()
        self.current_user = None
        self.glucose_data = None
        self.mcp_client = MistralMCPClient(api_key="ZAjtPftvZrCxK7WWwjBJIYudaiNhwRuO")
        self.chat_history = []

    def select_demo_user(self, user_key: str) -> Tuple[str, str, str]:
        """Handle demo user selection and simulate login"""
        if user_key not in DEMO_USERS:
            return gr.Info("❌ Invalid user selection"), "", ""

        try:
            user = DEMO_USERS[user_key]
            access_token = self.dexcom_api.simulate_demo_login(user_key)
            self.current_user = user

            # Initialize MCP client with user data
            asyncio.create_task(self.mcp_client.load_user_data(user_key))

            self.chat_history = []
            gr.Info(f"✅ Successfully connected as {user.name}")

            return f"Connected: {user.name} ({user.device_type})", f"Ready to chat with {user.name}", gr.update(visible=True)

        except Exception as e:
            gr.Error(f"❌ Connection failed: {str(e)}")
            return "", "", gr.update(visible=False)

    def load_glucose_data(self) -> Tuple[str, str, go.Figure]:
        """Load glucose data for the current user"""
        if not self.current_user:
            gr.Warning("Please select a demo user first")
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
                gr.Warning("No glucose data available")
                return "No glucose data available", "", None

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

            gr.Info("✅ Data loaded successfully!")
            return data_summary, "✅ Data loaded - ready for chat", chart

        except Exception as e:
            gr.Error(f"Failed to load glucose data: {str(e)}")
            return f"Failed to load glucose data: {str(e)}", "", None

    def get_template_prompts(self) -> List[str]:
        """Get template prompts based on user data"""
        if not self.current_user:
            return [
                "What should I know about managing my diabetes?",
                "How can I improve my glucose control?"
            ]

        if self.mcp_client.current_stats:
            stats = self.mcp_client.current_stats
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
            "Can you analyze my recent glucose patterns?",
            "What can I do to improve my diabetes management?"
        ]

    def chat_with_mistral(self, message: str, history: List) -> Tuple[str, List]:
        """Handle chat interaction with Mistral MCP"""
        if not message.strip():
            return "", history

        if not self.current_user:
            response = "Please select a demo user first to get personalized insights about glucose data."
            history.append([message, response])
            return "", history

        try:
            # Use asyncio to handle the async MCP client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.mcp_client.send_message(message))
            loop.close()

            history.append([message, response])
            return "", history

        except Exception as e:
            error_response = f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
            history.append([message, error_response])
            return "", history

    def use_template_prompt(self, template_text: str) -> str:
        """Use a template prompt in the chat"""
        return template_text

    def create_glucose_chart(self) -> Optional[go.Figure]:
        """Create an interactive glucose chart"""
        if self.glucose_data is None or self.glucose_data.empty:
            return None

        fig = go.Figure()

        colors = []
        for value in self.glucose_data['value']:
            if value < 70:
                colors.append('#E74C3C')
            elif value > 180:
                colors.append('#F39C12')
            else:
                colors.append('#27AE60')

        fig.add_trace(go.Scatter(
            x=self.glucose_data['systemTime'],
            y=self.glucose_data['value'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4, color=colors),
            hovertemplate='<b>%{y} mg/dL</b><br>%{x}<extra></extra>'
        ))

        fig.add_hrect(
            y0=70, y1=180,
            fillcolor="rgba(39, 174, 96, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="Target Range",
            annotation_position="top left"
        )

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

        for i in range(288 * 14):
            timestamp = base_time + timedelta(minutes=i * 5)

            hour = timestamp.hour

            base_glucose = 120 + 20 * np.sin((hour - 6) * np.pi / 12)

            if hour in [7, 12, 18]:
                base_glucose += random.randint(20, 60)

            glucose = base_glucose + random.randint(-15, 15)
            glucose = max(50, min(300, glucose))

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
    """

    with gr.Blocks(
        title="GlycoAI - AI Glucose Insights",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:

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

        with gr.Row():
            connection_status = gr.Textbox(
                label="Current User",
                value="No user selected",
                interactive=False,
                container=True
            )

        with gr.Group(visible=False) as main_interface:

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

            with gr.Tabs():

                with gr.TabItem("📈 Glucose Chart"):
                    glucose_chart = gr.Plot(
                        label="Interactive Glucose Trends",
                        container=True
                    )

                with gr.TabItem("💬 Chat with AI"):
                    gr.Markdown("### Chat with GlycoAI about your glucose data")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**💡 Quick Start Templates:**")
                            with gr.Row():
                                template1_btn = gr.Button(
                                    "🎯 Time in Range Help",
                                    variant="secondary",
                                    size="sm"
                                )
                                template2_btn = gr.Button(
                                    "⚡ Prevent Low Glucose",
                                    variant="secondary",
                                    size="sm"
                                )

                    chatbot = gr.Chatbot(
                        label="💬 Chat with GlycoAI",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )

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

                    with gr.Row():
                        clear_chat_btn = gr.Button(
                            "🗑️ Clear Chat",
                            variant="secondary",
                            size="sm"
                        )

                with gr.TabItem("📋 Data Overview"):
                    data_display = gr.Markdown("Load data to see overview", container=True)

        # Event handler functions
        def handle_user_selection(user_key):
            status, data_status, interface_visibility = app.select_demo_user(user_key)
            return status, data_status, interface_visibility, []

        def use_template_1():
            templates = app.get_template_prompts()
            return templates[0] if templates else "My time in range needs improvement. What specific strategies can help?"

        def use_template_2():
            templates = app.get_template_prompts()
            return templates[1] if len(templates) > 1 else "What are the best practices for preventing hypoglycemia?"

        def handle_chat_submit(message, history):
            return app.chat_with_mistral(message, history)

        def clear_chat():
            app.chat_history = []
            return []

        # Event handlers
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

        load_data_btn.click(
            app.load_glucose_data,
            outputs=[data_display, data_status, glucose_chart]
        )

# Chat event handlers (completion from where it was cut off)
        send_btn.click(
            handle_chat_submit,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )

        # Template button handlers
        template1_btn.click(
            use_template_1,
            outputs=[chat_input]
        )

        template2_btn.click(
            use_template_2,
            outputs=[chat_input]
        )

        # Clear chat handler
        clear_chat_btn.click(
            clear_chat,
            outputs=[chatbot]
        )

        # Add footer information
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
            share=False,            # Set to True for public sharing
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