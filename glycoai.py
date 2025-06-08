#!/usr/bin/env python3
"""
GlucoBuddy Main Application
Integrated application for glucose monitoring with Mistral chatbot
"""

import logging
import gradio as gr
from mistral_chat import GlucoBuddyMistralChat
from apifunctions import DexcomAPI, GlucoseAnalyzer, DEMO_USERS
from typing import Dict, Any

# Initialize Mistral chat with your agent ID
MISTRAL_API_KEY = "ZAjtPftvZrCxK7WWwjBJIYudaiNhwRuO"
MISTRAL_AGENT_ID = "ag:2d7a33b1:20250608:glycoaiagent:cc72ded9"

# Initialize the chat interface
mistral_chat = GlucoBuddyMistralChat(MISTRAL_API_KEY, MISTRAL_AGENT_ID)

# Initialize Dexcom API and Glucose Analyzer
dexcom_api = DexcomAPI()
analyzer = GlucoseAnalyzer()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the loaded data
loaded_data = None

def glucobuddy(user_query: str, history: list) -> str:
    """
    Handle user queries and generate responses using Mistral chatbot.
    """
    try:
        # Include loaded data in the user query context
        data_context = f"User Data: {loaded_data}" if loaded_data is not None else "No user data loaded."
        augmented_query = f"{user_query}\n\nContext: {data_context}"
        response = mistral_chat.chat_with_mistral(augmented_query)
        return response
    except Exception as e:
        logger.error(f"Error in glucobuddy function: {e}")
        return f"An error occurred: {e}"

def load_user_data(user_key: str) -> Dict[str, Any]:
    """
    Load glucose data for a specific demo user.
    """
    global loaded_data
    try:
        if user_key not in DEMO_USERS:
            return {"success": False, "message": f"Invalid user key. Available: {', '.join(DEMO_USERS.keys())}"}

        dexcom_api.simulate_demo_login(user_key)
        loaded_data = dexcom_api.get_glucose_data()
        return {"success": True, "glucose_data": loaded_data}
    except Exception as e:
        logger.error(f"Error loading user data: {e}")
        return {"success": False, "message": f"An error occurred: {e}"}

def create_interface():
    """
    Create and configure the Gradio interface.
    """
    with gr.Blocks() as interface:
        gr.Markdown("""
        <div style='text-align: center; max-width: 500px; margin: 0 auto;'>
            <h1>🌟 GlycoAI - AI-Powered Glucose Insights</h1>
            <p>Welcome to GlycoAI! Get personalized insights and recommendations for your glucose data.</p>
        </div>
        """)

        with gr.Tab("Data Analysis"):
            gr.Markdown("Visualize and analyze your glucose data")
            # Add your data analysis components here

        with gr.Tab("User Authentication"):
            gr.Markdown("Authenticate and load user data")
            user_key_input = gr.Textbox(label="Enter User Key")
            load_button = gr.Button("Load Data")
            data_output = gr.DataFrame(label="Glucose Data")
            load_button.click(
                fn=load_user_data,
                inputs=user_key_input,
                outputs=data_output
            )

        with gr.Tab("AI Chatbot"):
            gr.Markdown("Chat with your AI glucose assistant")
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(label="Your Message")
            user_input.submit(glucobuddy, inputs=user_input, outputs=chatbot)

        gr.Markdown("""
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-size: 0.9rem;'>
                ⚠️ Disclaimer: This tool is for informational and educational purposes only. Always consult your healthcare provider
                before making any changes to your diabetes management plan. This tool does not replace professional medical advice.</p>
            <p style='margin-top: 1rem; font-size: 0.9rem;'>
                🔒 Your data is processed securely and not stored permanently.
                💡 Powered by Dexcom API integration and Mistral AI.
            </p>
        </div>
        """)

    return interface

def main():
    """
    Main function to launch the application
    """
    print("🚀 Starting GlycoAI - AI-Powered Glucose Insights...")

    try:
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True,
            show_error=True,
            auth=None,
            favicon_path=None,
            ssl_verify=False
        )
    except Exception as e:
        logger.error(f"Failed to launch GlycoAI application: {e}")
        print(f"❌ Error launching application: {e}")
        raise

if __name__ == "__main__":
    main()
