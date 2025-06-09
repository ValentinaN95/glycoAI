---
title: GlycoAI - AI Glucose Insights
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: main.py
pinned: false
license: mit
tags:
  - healthcare
  - diabetes
  - glucose
  - ai-assistant
  - medical
  - gradio
  - mistral
  - agent-demo-track
  - dexcom
  - cgm
---

# 🩺 GlycoAI - AI-Powered Glucose Insights

**An intelligent diabetes management assistant powered by Mistral AI and Dexcom CGM integration**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange)](https://gradio.app/)
[![Mistral AI](https://img.shields.io/badge/Mistral%20AI-Agent-red)](https://mistral.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

GlycoAI is an advanced AI-powered chatbot that provides personalized glucose management insights for people with diabetes. By integrating Dexcom CGM data with Mistral AI's intelligent agents, it offers real-time analysis, pattern recognition, and actionable recommendations for better diabetes management.

### ✨ Key Features

- 📊 **14-Day Glucose Analysis**: Comprehensive pattern analysis across two weeks of data
- 🤖 **AI-Powered Insights**: Mistral AI agent provides personalized recommendations  
- 📈 **Interactive Visualizations**: Beautiful glucose trend charts and statistics
- 👥 **Demo Users**: Pre-configured profiles for different diabetes scenarios
- 💬 **Natural Conversations**: Chat naturally about your glucose patterns and concerns
- 🎯 **Clinical Targets**: Track time-in-range, hypoglycemia, and variability metrics
- 📱 **Multi-Device Support**: Works with G6, G7, and ONE+ CGM systems

## 🚀 Try It Now

**No setup required!** Simply:
1. Select a demo user (Sarah, Marcus, Jennifer, or Robert)
2. Load their 14-day glucose data 
3. Start chatting with GlycoAI about patterns and recommendations

## 🔬 Technical Implementation

### AI Agent Architecture
- **Mistral AI Agent**: Custom-trained agent specialized in diabetes management
- **Context-Aware**: Incorporates real glucose data into conversations
- **Pattern Recognition**: Identifies trends, meal effects, and lifestyle correlations
- **Personalized Advice**: Tailored recommendations based on individual patterns

### Data Processing Pipeline
```
Dexcom API → Data Validation → Pattern Analysis → AI Context → Chat Response
```

### Key Metrics Analyzed
- **Time in Range (TIR)**: Target 70-180 mg/dL
- **Glucose Variability**: Coefficient of variation
- **Hypoglycemia Risk**: Time below 70 mg/dL  
- **Hyperglycemia**: Time above 180 mg/dL
- **Daily Patterns**: Dawn phenomenon, meal effects
- **Weekly Trends**: Weekday vs weekend variations

## 👥 Demo Users

### 🏃‍♀️ Sarah Thompson (G7 Mobile)
- **Profile**: 32-year-old professional with Type 1 diabetes
- **Pattern**: Stable control with meal spikes
- **Device**: Dexcom G7 with smartphone integration

### 👨‍👧‍👦 Marcus Rodriguez (ONE+ Mobile) 
- **Profile**: 45-year-old father with Type 2 diabetes
- **Pattern**: Dawn phenomenon, moderate variability
- **Device**: Dexcom ONE+ with lifestyle management focus

### 🎓 Jennifer Chen (G6 Mobile)
- **Profile**: 28-year-old graduate student with Type 1 diabetes  
- **Pattern**: Exercise-related lows, tech-savvy user
- **Device**: Dexcom G6 with active lifestyle

### 👨‍🏫 Robert Williams (G6 Receiver)
- **Profile**: 67-year-old retired teacher with Type 2 diabetes
- **Pattern**: Consistent dawn phenomenon
- **Device**: Dexcom G6 with dedicated receiver

## 💡 Example Conversations

**"What's my average glucose level?"**
> Based on your 14-day data, your average glucose is 142 mg/dL with good stability. Your time in range is 68%, which is close to the clinical target of >70%. 📊

**"I keep having morning highs. What can I do?"**
> I notice you have dawn phenomenon with glucose rising 30-40 mg/dL between 4-7 AM. This affects 5 out of 14 mornings in your data. Consider discussing overnight insulin adjustments with your healthcare provider. 🌅

**"How does my weekend compare to weekdays?"**
> Interesting pattern! Your weekends show 15 mg/dL lower average glucose (135 vs 150 mg/dL weekdays). You seem to have more consistent meal timing on weekends. 📈

## 🛠️ Technical Stack

- **Frontend**: Gradio 4.44.0 with custom CSS styling
- **Backend**: Python with FastAPI-style architecture  
- **AI Engine**: Mistral AI agents with specialized diabetes knowledge
- **Data Source**: Dexcom Sandbox API with realistic mock data
- **Visualization**: Plotly for interactive glucose charts
- **Processing**: Pandas/NumPy for statistical analysis

## 📊 Data Analysis Features

### Pattern Recognition
- **Meal Effects**: Identifies post-meal glucose spikes and timing
- **Exercise Impact**: Detects glucose drops during physical activity  
- **Sleep Patterns**: Analyzes overnight glucose stability
- **Stress Correlation**: Identifies high-glucose periods linked to lifestyle

### Statistical Analysis  
- **Glucose Management Indicator (GMI)**: HbA1c estimation
- **Coefficient of Variation**: Glucose stability measurement
- **Time-in-Range Analysis**: Clinical target tracking
- **Trend Analysis**: Week-over-week improvement detection

### Predictive Insights
- **Risk Identification**: Predicts hypoglycemia patterns
- **Optimization Suggestions**: Recommends timing adjustments
- **Lifestyle Correlations**: Links patterns to daily activities
- **Goal Tracking**: Monitors progress toward clinical targets

## 🔒 Privacy & Security

- **No Data Storage**: Conversations and glucose data are not permanently stored
- **Sandbox Environment**: Uses Dexcom's secure sandbox API
- **Educational Purpose**: Designed for demonstration and learning
- **Medical Disclaimer**: Not intended to replace professional medical advice

## ⚕️ Medical Disclaimer

**Important**: GlycoAI is for educational and informational purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers before making any changes to your diabetes management plan.

## 🎯 Agent Demo Track

This application showcases advanced AI agent capabilities for healthcare applications:

- **Contextual Understanding**: Processes complex medical data
- **Personalized Responses**: Adapts advice to individual patterns  
- **Multi-Modal Analysis**: Combines numerical data with conversational AI
- **Domain Expertise**: Specialized knowledge in diabetes management
- **Real-Time Processing**: Instant analysis of glucose trends

Perfect example of how AI agents can augment healthcare decision-making while maintaining appropriate clinical boundaries.

## 🚀 Getting Started

### Option 1: Use This Space (Recommended)
Just click the demo above! No installation needed.

### Option 2: Local Installation
```bash
git clone https://github.com/your-repo/glycoai
cd glycoai
pip install -r requirements.txt
python main.py
```

### Option 3: API Integration
```python
from mistral_chat import GlucoBuddyMistralChat

# Initialize with your Mistral API key
chat = GlucoBuddyMistralChat("your-api-key", "your-agent-id")

# Load demo user
chat.load_user_data("sarah_g7")

# Start chatting
response = chat.chat_with_mistral("What's my time in range?")
print(response['response'])
```

## 📈 Roadmap

- [ ] **Real Dexcom Integration**: Connect to live CGM data
- [ ] **Health data**: integration with Apple Health for obtaining further insights (e.g hormonal influence)
- [ ] **Insulin Tracking**: Dosing recommendations and timing
- [ ] **Healthcare Provider Dashboard**: Shareable reports
- [ ] **Mobile App**: Native iOS/Android applications
- [ ] **Multiple Languages**: Multilingual diabetes support

## 🤝 Contributing

We welcome contributions! Areas of interest:
- **Medical Accuracy**: Improve clinical recommendations
- **UI/UX Enhancement**: Better user experience design  
- **Data Analysis**: Advanced pattern recognition algorithms
- **Agent Training**: Enhance AI conversation quality
- **Integration**: Additional CGM device support

## 📞 Support

- **Documentation**: [Full Documentation](https://github.com/your-repo/glycoai/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-repo/glycoai/issues)
- **Discussions**: [Community Forum](https://github.com/your-repo/glycoai/discussions)



## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the diabetes community**

*Empowering better glucose management through AI*

[![Follow on HF](https://img.shields.io/badge/Follow%20on-Hugging%20Face-yellow)](https://huggingface.co/spaces/)
[![Star on GitHub](https://img.shields.io/badge/Star%20on-GitHub-black)](https://github.com/your-repo/glycoai)

</div>