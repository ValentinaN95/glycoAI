"""
GlycoAI - AI-Powered Glucose Insights
Main Gradio application with prominent, centralized load data button
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Tuple, List
import logging
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import the Mistral chat class and unified data manager
from mistral_chat import GlucoBuddyMistralChat, validate_environment
from unified_data_manager import UnifiedDataManager

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
    """Main application class for GlucoBuddy with unified data management"""

    def __init__(self):
        # Validate environment before initializing
        if not validate_environment():
            raise ValueError("Environment validation failed - check your .env file or environment variables")
        
        # Single data manager for consistency
        self.data_manager = UnifiedDataManager()
        
        # Chat interface (will use data manager's context)
        self.mistral_chat = GlucoBuddyMistralChat()
        
        # UI state
        self.chat_history = []

    def select_demo_user(self, user_key: str) -> Tuple[str, str]:
        """Handle demo user selection and load data consistently"""
        if user_key not in DEMO_USERS:
            return "❌ Invalid user selection", gr.update(visible=False)

        try:
            # Load data through unified manager
            load_result = self.data_manager.load_user_data(user_key)
            
            if not load_result['success']:
                return f"❌ {load_result['message']}", gr.update(visible=False)
            
            user = self.data_manager.current_user
            
            # Update Mistral chat with the same context
            self._sync_chat_with_data_manager()
            
            # Clear chat history when switching users
            self.chat_history = []
            self.mistral_chat.clear_conversation()

            return (
                f"Connected: {user.name} ({user.device_type}) - Click 'Load Data' to begin", 
                gr.update(visible=True)
            )

        except Exception as e:
            logger.error(f"User selection failed: {str(e)}")
            return f"❌ Connection failed: {str(e)}", gr.update(visible=False)

    def load_glucose_data(self) -> Tuple[str, go.Figure]:
        """Load and display glucose data using unified manager"""
        if not self.data_manager.current_user:
            return "Please select a demo user first", None

        try:
            # Force reload data to ensure freshness
            load_result = self.data_manager.load_user_data(
                self._get_current_user_key(), 
                force_reload=True
            )
            
            if not load_result['success']:
                return load_result['message'], None
            
            # Get unified stats
            stats = self.data_manager.get_stats_for_ui()
            chart_data = self.data_manager.get_chart_data()
            
            # Sync chat with fresh data
            self._sync_chat_with_data_manager()
            
            if chart_data is None or chart_data.empty:
                return "No glucose data available", None

            # Build data summary with CONSISTENT metrics
            user = self.data_manager.current_user
            data_points = stats.get('total_readings', 0)
            avg_glucose = stats.get('average_glucose', 0)
            std_glucose = stats.get('std_glucose', 0)
            min_glucose = stats.get('min_glucose', 0)
            max_glucose = stats.get('max_glucose', 0)
            
            time_in_range = stats.get('time_in_range_70_180', 0)
            time_below_range = stats.get('time_below_70', 0)
            time_above_range = stats.get('time_above_180', 0)
            
            gmi = stats.get('gmi', 0)
            cv = stats.get('cv', 0)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)

            data_summary = f"""
## 📊 Data Summary for {user.name}

### Basic Information
• **Analysis Period:** {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')} (14 days)
• **Total Readings:** {data_points:,} glucose measurements
• **Device:** {user.device_type}
• **Data Source:** {stats.get('data_source', 'unknown').upper()}

### Glucose Statistics
• **Average Glucose:** {avg_glucose:.1f} mg/dL
• **Standard Deviation:** {std_glucose:.1f} mg/dL
• **Coefficient of Variation:** {cv:.1f}%
• **Glucose Range:** {min_glucose:.0f} - {max_glucose:.0f} mg/dL
• **GMI (Glucose Management Indicator):** {gmi:.1f}%

### Time in Range Analysis
• **Time in Range (70-180 mg/dL):** {time_in_range:.1f}%
• **Time Below Range (<70 mg/dL):** {time_below_range:.1f}%
• **Time Above Range (>180 mg/dL):** {time_above_range:.1f}%

### Clinical Targets
• **Target Time in Range:** >70% (Current: {time_in_range:.1f}%)
• **Target Time Below Range:** <4% (Current: {time_below_range:.1f}%)
• **Target CV:** <36% (Current: {cv:.1f}%)

### Data Validation
• **In Range Count:** {stats.get('in_range_count', 0)} readings
• **Below Range Count:** {stats.get('below_range_count', 0)} readings
• **Above Range Count:** {stats.get('above_range_count', 0)} readings
• **Total Verified:** {stats.get('in_range_count', 0) + stats.get('below_range_count', 0) + stats.get('above_range_count', 0)} readings

### 14-Day Analysis Benefits
• **Enhanced Pattern Recognition:** Captures full weekly cycles and variations
• **Improved Trend Analysis:** Identifies consistent patterns vs. one-time events
• **Better Clinical Insights:** More reliable data for healthcare decisions
• **AI Consistency:** Same data used for chat analysis and UI display
            """

            chart = self.create_glucose_chart()
            
            return data_summary, chart

        except Exception as e:
            logger.error(f"Failed to load glucose data: {str(e)}")
            return f"Failed to load glucose data: {str(e)}", None

    def _sync_chat_with_data_manager(self):
        """Ensure chat uses the same data as the UI"""
        try:
            # Get context from unified data manager
            context = self.data_manager.get_context_for_agent()
            
            # Update chat's internal data to match
            if not context.get("error"):
                self.mistral_chat.current_user = self.data_manager.current_user
                self.mistral_chat.current_glucose_data = self.data_manager.processed_glucose_data
                self.mistral_chat.current_stats = self.data_manager.calculated_stats
                self.mistral_chat.current_patterns = self.data_manager.identified_patterns
                
                logger.info(f"Synced chat with data manager - TIR: {self.data_manager.calculated_stats.get('time_in_range_70_180', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to sync chat with data manager: {e}")

    def _get_current_user_key(self) -> str:
        """Get the current user key"""
        if not self.data_manager.current_user:
            return ""
        
        # Find the key for current user
        for key, user in DEMO_USERS.items():
            if user == self.data_manager.current_user:
                return key
        return ""

    def get_template_prompts(self) -> List[str]:
        """Get template prompts based on current user data"""
        if not self.data_manager.current_user or not self.data_manager.calculated_stats:
            return [
                "What should I know about managing my diabetes?",
                "How can I improve my glucose control?"
            ]

        stats = self.data_manager.calculated_stats
        time_in_range = stats.get('time_in_range_70_180', 0)
        time_below_70 = stats.get('time_below_70', 0)

        templates = []

        if time_in_range < 70:
            templates.append(f"My time in range is {time_in_range:.1f}% which is below the 70% target. What specific strategies can help me improve it?")
        else:
            templates.append(f"My time in range is {time_in_range:.1f}% which meets the target. How can I maintain this level of control?")

        if time_below_70 > 4:
            templates.append(f"I'm experiencing {time_below_70:.1f}% time below 70 mg/dL. What can I do to prevent these low episodes?")
        else:
            templates.append("What are the best practices for preventing hypoglycemia in my situation?")

        return templates

    def chat_with_mistral(self, message: str, history: List) -> Tuple[str, List]:
        """Handle chat interaction with Mistral using unified data"""
        if not message.strip():
            return "", history

        if not self.data_manager.current_user:
            response = "Please select a demo user first to get personalized insights about glucose data."
            history.append([message, response])
            return "", history

        try:
            # Ensure chat is synced with latest data
            self._sync_chat_with_data_manager()
            
            # Send message to Mistral chat
            result = self.mistral_chat.chat_with_mistral(message)
            
            if result['success']:
                response = result['response']
                
                # Add data consistency note
                validation = self.data_manager.validate_data_consistency()
                if validation.get('valid'):
                    data_age = validation.get('data_age_minutes', 0)
                    if data_age > 10:  # Warn if data is old
                        response += f"\n\n📊 *Note: Analysis based on data from {data_age} minutes ago. Reload data for most current insights.*"
                
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
        """Create an interactive glucose chart using unified data"""
        chart_data = self.data_manager.get_chart_data()
        
        if chart_data is None or chart_data.empty:
            return None

        fig = go.Figure()

        # Color code based on glucose ranges
        colors = []
        for value in chart_data['value']:
            if value < 70:
                colors.append('#E74C3C')  # Red for low
            elif value > 180:
                colors.append('#F39C12')  # Orange for high
            else:
                colors.append('#27AE60')  # Green for in range

        fig.add_trace(go.Scatter(
            x=chart_data['systemTime'],
            y=chart_data['value'],
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

        # Get current stats for title
        stats = self.data_manager.get_stats_for_ui()
        tir = stats.get('time_in_range_70_180', 0)
        
        fig.update_layout(
            title={
                'text': f"14-Day Glucose Trends - {self.data_manager.current_user.name} (TIR: {tir:.1f}%)",
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


def create_interface():
    """Create the Gradio interface with prominent, centralized load data button"""
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
    .load-data-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .prominent-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1.5rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        color: white !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        min-height: 80px !important;
        text-align: center !important;
    }
    .prominent-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.6) !important;
    }
    .data-status-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: 500;
    }
    .data-status-success {
        border-color: #28a745;
        background: #d4edda;
        color: #155724;
    }
    .data-status-error {
        border-color: #dc3545;
        background: #f8d7da;
        color: #721c24;
    }
    .user-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
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
    .section-divider {
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1px;
        margin: 2rem 0;
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

        # Section Divider
        gr.HTML('<div class="section-divider"></div>')

        # PROMINENT CENTRALIZED DATA LOADING SECTION
        with gr.Group(visible=False) as main_interface:
            # PROMINENT LOAD BUTTON - Centered and Large
            with gr.Row():
                with gr.Column(scale=1):
                    pass  # Left spacer
                with gr.Column(scale=2):
                    load_data_btn = gr.Button(
                        "🚀 LOAD 14-DAY GLUCOSE DATA\n📈 Start Analysis & Enable AI Chat",
                        elem_classes=["prominent-button"],
                        size="lg"
                    )
                with gr.Column(scale=1):
                    pass  # Right spacer

            # Section Divider
            gr.HTML('<div class="section-divider"></div>')

            # Main Content Tabs
            with gr.Tabs():

                # Glucose Chart Tab
                with gr.TabItem("📈 Glucose Chart"):
                    with gr.Column():
                        gr.Markdown("### 📊 Interactive 14-Day Glucose Analysis")
                        gr.Markdown("*Load your data using the button above to see your comprehensive glucose trends*")
                        
                        glucose_chart = gr.Plot(
                            label="Interactive 14-Day Glucose Trends",
                            container=True
                        )

                # Chat Tab
                with gr.TabItem("💬 Chat with AI"):
                    with gr.Column(elem_classes=["chat-container"]):
                        gr.Markdown("### 🤖 Chat with GlycoAI about your glucose data")
                        gr.Markdown("*📊 Load your data using the button above to enable personalized AI insights*")
                        
                        # Template Prompts
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**💡 Quick Start Templates:**")
                                with gr.Row():
                                    template1_btn = gr.Button(
                                        "🎯 Analyze My 14-Day Patterns",
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
                            label="💬 Chat with GlycoAI (Unified Data)",
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
                    with gr.Column():
                        gr.Markdown("### 📋 Comprehensive Data Analysis")
                        gr.Markdown("*Load your data using the button above to see detailed glucose statistics*")
                        
                        data_display = gr.Markdown("Click 'Load 14-Day Glucose Data' above to see your comprehensive analysis", container=True)

        # Event Handlers
        def handle_user_selection(user_key):
            status, interface_visibility = app.select_demo_user(user_key)
            return status, interface_visibility, []

        def handle_load_data():
            overview, status_overview, chart, status_chart = app.load_glucose_data()
            return overview, chart

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
        user_selection_outputs = [connection_status, main_interface, chatbot]

        sarah_btn.click(
            lambda: handle_user_selection("sarah_g7"),
            outputs=[connection_status, main_interface, chatbot]
        )

        marcus_btn.click(
            lambda: handle_user_selection("marcus_one"),
            outputs=[connection_status, main_interface, chatbot]
        )

        jennifer_btn.click(
            lambda: handle_user_selection("jennifer_g6"),
            outputs=[connection_status, main_interface, chatbot]
        )

        robert_btn.click(
            lambda: handle_user_selection("robert_receiver"),
            outputs=[connection_status, main_interface, chatbot]
        )

        # PROMINENT DATA LOADING - Single button updates all views
        load_data_btn.click(
            handle_load_data,
            outputs=[data_display, glucose_chart]
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
    print("🚀 Starting GlycoAI - AI-Powered Glucose Insights (Enhanced UI)...")
    
    # Validate environment before starting
    print("🔍 Validating environment configuration...")
    if not validate_environment():
        print("❌ Environment validation failed!")
        print("Please check your .env file or environment variables.")
        return
    
    print("✅ Environment validation passed!")
    
    try:
        # Create and launch the interface
        demo = create_interface()
        
        print("🎯 GlycoAI is starting with enhanced UI design...")
        print("📊 Features: Prominent load button, unified data management, consistent metrics")
        
        # Launch with custom settings
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default Gradio port
            share=True,            # Set to True for public sharing (tunneling)
            debug=os.getenv("DEBUG", "false").lower() == "true",
            show_error=True,        # Show errors in the interface
            auth=None,              # No authentication required
            favicon_path=None,      # Use default favicon
            ssl_verify=False        # Disable SSL verification for development
        )
        
    except Exception as e:
        logger.error(f"Failed to launch GlycoAI application: {e}")
        print(f"❌ Error launching application: {e}")
        
        # Provide helpful error information
        if "environment" in str(e).lower():
            print("\n💡 Environment troubleshooting:")
            print("1. Check if .env file exists with MISTRAL_API_KEY")
            print("2. Verify your API key is valid")
            print("3. For Hugging Face Spaces, check Repository secrets")
        else:
            print("\n💡 Try checking:")
            print("1. All dependencies are installed: pip install -r requirements.txt")
            print("2. Port 7860 is available")
            print("3. Check the logs above for specific error details")
        
        raise


if __name__ == "__main__":
    # Setup logging configuration
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('glycoai.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run the main application
    main()