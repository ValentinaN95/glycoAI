# 🩺 GlucoBuddy - AI-Powered Glucose Insights

GlucoBuddy is an intelligent application that analyzes continuous glucose monitoring (CGM) data from Dexcom devices and provides personalized insights using AI-powered analysis through Claude MCP (Model Context Protocol).

## ✨ Features

- **Multi-Device Support**: Works with G6, G7, Dexcom ONE, and ONE+ devices
- **AI-Powered Analysis**: Uses Claude MCP for intelligent pattern recognition
- **Interactive Dashboard**: Beautiful Gradio interface with real-time charts
- **Demo Mode**: Test with realistic simulated data from 4 different user profiles
- **Comprehensive Insights**: Time-in-range analysis, pattern detection, and personalized recommendations
- **HIPAA-Aware**: Designed with healthcare data privacy in mind

## 🏗️ Architecture

The application is built with a modular architecture:

- `glucobuddy_functions.py` - Core API integrations and data analysis
- `main.py` - Gradio interface and application logic
- `config.py` - Configuration management and API keys
- `requirements.txt` - Python dependencies

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Dexcom Developer Account (for production use)
- Anthropic API Key (for AI insights)

### Installation

1. **Clone or download the application files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your actual credentials
   ```

4. **Set up Dexcom API credentials** (optional for demo):
   - Register at [Dexcom Developer Portal](https://developer.dexcom.com)
   - Create a new application
   - Add your CLIENT_ID and CLIENT_SECRET to .env

5. **Set up Claude API** (optional for enhanced insights):
   - Get API key from [Anthropic Console](https://console.anthropic.com)
   - Add ANTHROPIC_API_KEY to .env

### Running the Application

```bash
python main.py
```

The application will start on `http://localhost:7860`

## 👥 Demo Users

GlucoBuddy includes 4 realistic demo users for testing:

| User | Device | Profile |
|------|--------|---------|
| **Sarah Thompson** | G7 Mobile App | 32-year-old professional with Type 1 diabetes |
| **Marcus Rodriguez** | ONE+ Mobile App | 45-year-old father with Type 2 diabetes |
| **Jennifer Chen** | G6 Mobile App | 28-year-old graduate student with Type 1 diabetes |
| **Robert Williams** | G6 Receiver | 67-year-old retiree with Type 2 diabetes |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEXCOM_CLIENT_ID` | Dexcom API Client ID | For production |
| `DEXCOM_CLIENT_SECRET` | Dexcom API Client Secret | For production |
| `ANTHROPIC_API_KEY` | Claude API Key | For AI insights |
| `DEBUG` | Enable debug mode | No |
| `HOST` | Application host | No |
| `PORT` | Application port | No |

### Glucose Analysis Settings

- **Target Range**: 70-180 mg/dL (configurable)
- **Severe Hypoglycemia**: <54 mg/dL
- **Severe Hyperglycemia**: >250 mg/dL
- **Default Analysis Period**: 7 days

## 📊 Features Overview

### 1. User Authentication
- OAuth 2.0 integration with Dexcom API
- Demo mode with 4 pre-configured users
- Secure token management

### 2. Data Analysis
- **Time in Range**: Percentage of readings in target range
- **Glucose Variability**: Standard deviation analysis
- **Pattern Recognition**: Daily patterns, meal responses, exercise effects
- **Risk Assessment**: Hypoglycemia and hyperglycemia detection

### 3. AI
