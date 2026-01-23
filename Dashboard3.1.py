import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta, date
import time
import google.generativeai as genai
import os

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="GPF Biogas Dashboard",
    # page_icon= ,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================
BASE_URL = "http://127.0.0.1:8000"
GEMINI_API_KEY = "AIzaSyCRLv4DyyFKrovknk06A9p69dN_RG1BX0M"

# ==============================================================================
# STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .farmer-metric {
        background: linear-gradient(135deg, #32CD32 0%, #228B22 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .farmer-metric:hover {
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    .metric-unit {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .status-excellent { 
        background: linear-gradient(135deg, #32CD32 0%, #228B22 100%); 
    }
    .status-good { 
        background: linear-gradient(135deg, #90EE90 0%, #32CD32 100%); 
    }
    .status-warning { 
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
    }
    .status-alert { 
        background: linear-gradient(135deg, #FF6347 0%, #DC143C 100%); 
    }
    
    .info-box {
        background: linear-gradient(135deg, #F0F8FF 0%, #E6F3FF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4169E1;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .tip-box {
        background: linear-gradient(135deg, #F0FFF0 0%, #E6FFE6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #32CD32;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFEBCD 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF8C00;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 2px solid #E8F5E8;
    }
    
    .section-header {
        background: linear-gradient(135deg, #32CD32 0%, #228B22 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #F0F8F0 0%, #E8F5E8 100%);
    }
    
    .nav-button {
        background: linear-gradient(135deg, #32CD32 0%, #228B22 100%);
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 8px;
        border: none;
        margin: 0.3rem 0;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(50, 205, 50, 0.3);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        margin-top: 3rem;
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# AI ASSISTANT SETUP
# ==============================================================================
def setup_gemini():
    """Setup Google Gemini AI"""
    try:
        if not GEMINI_API_KEY:
            st.error("🤖 Please set GEMINI_API_KEY in configuration")
            return None
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"🤖 AI Agent setup error: {str(e)}")
        return None

def get_biogas_ai_response(model, user_question, farm_context=""):
    """Get AI response with biogas expertise"""
    if not model:
        return "🤖 AI Agent is currently unavailable. Please check your API configuration."
    
    system_prompt = f"""You are BioAIgent, an expert AI assistant specialising in biogas production for farmers. 
    You help farmers optimise their biogas digesters, troubleshoot issues, and maximise gas production.
    
    Key areas of expertise:
    - Optimal temperature ranges (35-40°C for mesophilic digesters)
    - pH balance (6.8-7.2 optimal range) 
    - Organic Loading Rate (OLR) optimisation
    - Feeding schedules and substrate preparation
    - Common problems and solutions
    - Safety protocols
    - Economic benefits and cost calculations
    - Seasonal management
    - System maintenance
    
    Always provide practical, actionable advice suitable for farmers in Southern Africa.
    Keep responses concise but informative. Use simple language and avoid overly technical jargon.
    Include specific numbers/ranges when relevant.
    
    Farm Context: {farm_context}
    
    Answer this farmer's question:"""
    
    try:
        full_prompt = f"{system_prompt}\n\nFarmer's Question: {user_question}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"🤖 Sorry, I couldn't process your question right now. Error: {str(e)}" #keep truck of the chat

# ==============================================================================
# AUTHENTICATION
# ==============================================================================
def check_credentials(username, password):
    return username == "Vasco" and password == "63072544"

def login():
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%); 
                 padding: 3rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin-bottom: 1rem;'>GPF Dashboard</h1>
        <p style='color: white; font-size: 1.2rem;'>Your AI Assistant for Biogas Production</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Login to Your Dashboard")
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        if st.button("Login", use_container_width=True):
            if check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Welcome back! Loading your dashboard...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

# ==============================================================================
# DATA FUNCTIONS
# ==============================================================================
@st.cache_data(ttl=300)
def fetch_data(date="2025-03-31"):
    try:
        with st.spinner("🔄 Getting your latest farm data..."):
            res = requests.post(f"{BASE_URL}/get_data", json={"date": date}, timeout=10)
            res.raise_for_status()
            return res.json()
    except Exception as e:
        st.error(f"📡 Connection issue: {str(e)}")
        return []

@st.cache_data(ttl=300)
def fetch_prediction(date="2025-08-05"):
    try:
        res = requests.post(f"{BASE_URL}/get_prediction", json={"date": date}, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"🔮 Prediction service unavailable: {str(e)}")
        return None

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = setup_gemini()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": " Hi! I'm BioAIgent, your AI assistant for biogas production. Ask me anything about optimising your biodigester, troubleshooting issues, or maximising biogas production!"}
    ]

if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = False

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
if not st.session_state.logged_in:
    login()
    st.stop()

# ==============================================================================
# HEADER
# ==============================================================================
col1, col2 = st.columns([1, 4])
with col1:
    try:
        st.image("GPFlogo.png", width=120)
    except:
        st.markdown(" ", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='text-align: right; padding: 1rem;'>
        <h2 style='color: #2E8B57; margin: 0;'>Welcome back, {st.session_state.username.capitalize()}!</h2>
        <p style='color: #666; margin: 0.5rem 0 0 0;'>Managing your biogas production smartly</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# AI CHAT INTERFACE
# ==============================================================================
def display_chat_interface():
    """Display the AI chat interface"""
    with st.sidebar:
        st.markdown("### ✨ AI Assistant")
        if st.button("💬 Chat with BioAIgent", use_container_width=True):
            st.session_state.chat_visible = not st.session_state.chat_visible
            st.rerun()
        
        if st.session_state.chat_visible:
            st.success(" BioAigent is active!")
            if st.button(" Close Chat", use_container_width=True):
                st.session_state.chat_visible = False
                st.rerun()
            
            # Display chat messages
            for message in st.session_state.chat_messages:
                icon = "👤" if message["role"] == "user" else "🤖"
                bg_color = "#e3f2fd" if message["role"] == "user" else "#f1f8e9"
                st.markdown(f"""
                <div style="margin-bottom: 1rem; padding: 0.8rem; border-radius: 10px; 
                           background: {bg_color};">
                    <strong>{icon}:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            # Chat input
            user_input = st.text_input("Ask BioAIgent...", key="chat_input", 
                                     placeholder="e.g., Why is my biogas production low?")
            
            if st.button("Send", key="send_chat", use_container_width=True):
                if user_input.strip():
                    # Add user message
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    
                    # Get farm context
                    try:
                        data = fetch_data()
                        if data:
                            df = pd.DataFrame(data)
                            farm_context = f"""
                            Current farm stats:
                            - Average biogas production: {df['Biogas Production (m³)'].mean():.6f} m³/day
                            - Average temperature: {df['Temperature (°C)'].mean():.1f}°C
                            - Average pH: {df['pH'].mean():.1f}
                            - Average OLR: {df['OLR (kg VS/m³/day)'].mean():.1f} kg VS/m³/day
                            """
                        else:
                            farm_context = "No current farm data available."
                    except:
                        farm_context = "Farm data temporarily unavailable."
                    
                    # Get AI response
                    with st.spinner("BioAIgent is thinking..."):
                        ai_response = get_biogas_ai_response(st.session_state.gemini_model, user_input, farm_context)
                    
                    # Add AI response
                    st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                    st.rerun()

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #32CD32 0%, #228B22 100%); 
           padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0;'>Menu</h3>
</div>
""", unsafe_allow_html=True)

nav_options = {
    "Biodigester Overview": "overview",
    "Advanced Analytics": "analytics", 
    "Production Forecast": "prediction",
    "Farm Insights": "insights",
    "System Diagrams": "diagrams",
    "Settings": "settings"
}

selected_nav = st.sidebar.radio("Navigate to:", list(nav_options.keys()))
app_mode = nav_options[selected_nav]

# Display AI Chat Interface
display_chat_interface()

# Date Controls
st.sidebar.markdown("### Data Controls")
date_input = st.sidebar.date_input(
    "Analysis Date", 
    value=pd.to_datetime("2025-03-31"),
    help="Select the date you want to analyze"
)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ==============================================================================
# PAGE CONTENT
# ==============================================================================

# BIODIGESTER OVERVIEW PAGE
if app_mode == "overview":
    st.markdown("""
    <div class="main-header">
        <h1>Biodigester Overview</h1>
        <p>Real-time insights to maximize your biogas production and save money</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch and process data
    data = fetch_data(str(date_input))
    
    if not data:
        st.markdown("""
        <div class="warning-box">
            <h3>📊 No Data Available</h3>
            <p>We couldn't fetch your farm data right now. Please check your internet connection or try again later.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Key Performance Metrics
    st.markdown('<div class="section-header"> Your Farm Performance Today</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_production = df['Biogas Production (m³)'].mean()
        status = "status-excellent" if avg_production > 35 else "status-good" if avg_production > 25 else "status-warning"
        st.markdown(f"""
        <div class="farmer-metric {status}">
            <div class="metric-value">{10000*avg_production:.2f}</div>
            <div class="metric-label">Daily Biogas</div>
            <div class="metric-unit">cubic meters (m³)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_production = df['Biogas Production (m³)'].sum()
        monthly_estimate = total_production * (30 / len(df))
        st.markdown(f"""
        <div class="farmer-metric status-good">
            <div class="metric-value">{10000*monthly_estimate:.2f}</div>
            <div class="metric-label">Monthly Estimate</div>
            <div class="metric-unit">cubic meters (m³)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_temp = df['Temperature (°C)'].mean()
        temp_status = "status-excellent" if 35 <= avg_temp <= 40 else "status-warning"
        st.markdown(f"""
        <div class="farmer-metric {temp_status}">
            <div class="metric-value">{avg_temp:.1f}°C</div>
            <div class="metric-label">System Temperature</div>
            <div class="metric-unit">optimal: 35-40°C</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        revenue_estimate = avg_production * 30 * 2.65
        st.markdown(f"""
        <div class="farmer-metric status-good">
            <div class="metric-value">{10000*revenue_estimate:.2f} Mts</div>
            <div class="metric-label">Monthly Savings</div>
            <div class="metric-unit">estimated earnings</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Production Chart
    st.markdown('<div class="section-header"> Your Biogas Production Trend</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>💡 What this chart tells you:</h4>
        <p><strong>Blue line:</strong> Your daily biogas production</p>
        <p><strong>Green dashed line:</strong> 7-day average (smooths out daily variations)</p>
        <p><strong>Why it matters:</strong> Shows if your production is increasing, stable, or decreasing over time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df_ts = df.set_index('Timestamp')
    df_ts['7d_avg'] = df_ts['Biogas Production (m³)'].rolling(7).mean()
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, 
        y=df_ts['Biogas Production (m³)'], 
        mode='lines+markers',
        name='Daily Production',
        line=dict(color='#4169E1', width=3),
        marker=dict(size=6, color='#4169E1')
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts.index, 
        y=df_ts['7d_avg'], 
        mode='lines',
        name='7-Day Average',
        line=dict(color='#32CD32', width=4, dash='dash')
    ))
    
    fig_ts.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Date",
        yaxis_title="Biogas Production (m³)",
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Performance Analysis
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown('<div class="section-header"> Performance Rating</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tip-box">
            <h4>🎯 Your Performance Gauge</h4>
            <p><strong>Green zone (35-50):</strong> Excellent production</p>
            <p><strong>Yellow zone (20-35):</strong> Good, room for improvement</p>
            <p><strong>Red zone (0-20):</strong> Needs attention</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_production,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Daily Average (m³)"},
            delta = {'reference': 10},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "#32CD32"},
                'steps': [
                    {'range': [0, 20], 'color': "#FFE4E1"},
                    {'range': [20, 35], 'color': "#FFFACD"},
                    {'range': [35, 50], 'color': "#F0FFF0"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col1:
        st.markdown('<div class="section-header"> Production Categories</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>📊 What this pie chart shows:</h4>
            <p><strong>Purpose:</strong> Breaks down your production days into Low, Medium, and High categories</p>
            <p><strong>Goal:</strong> You want more days in the "High" category (green slice)</p>
        </div>
        """, unsafe_allow_html=True)
        
        low_thresh = df['Biogas Production (m³)'].quantile(0.33)
        high_thresh = df['Biogas Production (m³)'].quantile(0.67)
        
        conditions = [
            df['Biogas Production (m³)'] <= low_thresh,
            (df['Biogas Production (m³)'] > low_thresh) & (df['Biogas Production (m³)'] <= high_thresh),
            df['Biogas Production (m³)'] > high_thresh
        ]
        categories = ['Low Production', 'Medium Production', 'High Production']
        df['Production_Category'] = np.select(conditions, categories, default='Unknown')
        
        category_counts = df['Production_Category'].value_counts()
        
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color_discrete_sequence=['#FF6B6B', '#FFD93D', '#4ECDC4'],
            title="Days by Production Level"
        )
        fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

# PRODUCTION ANALYTICS PAGE
elif app_mode == "analytics":
    st.markdown("""
    <div class="main-header">
        <h1> Production Analytics</h1>
        <p>Deep dive into your biogas production patterns and optimization opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date range selector
    st.markdown('<div class="section-header"> Select Analysis Period</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 3, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 3, 31))
    
    data = fetch_data(str(end_date))
    if not data:
        st.warning("⚠️ No data available for analysis.")
        st.stop()
    
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df_filtered = df[(df['Timestamp'] >= pd.to_datetime(start_date)) & 
                     (df['Timestamp'] <= pd.to_datetime(end_date))]
    
    # Temperature vs Production Analysis
    st.markdown('<div class="section-header"> Temperature Impact Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🌡️ Why temperature matters for your biogas production:</h4>
        <p><strong>Sweet spot:</strong> 35-40°C produces the most biogas</p>
        <p><strong>Too cold:</strong> Below 30°C - bacteria slow down, less gas production</p>
        <p><strong>Too hot:</strong> Above 45°C - bacteria die, production drops</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp_scatter = px.scatter(
            df_filtered, 
            x="Temperature (°C)", 
            y="Biogas Production (m³)",
            size="OLR (kg VS/m³/day)",
            color="pH",
            hover_data=["Timestamp"],
            title="Temperature vs Biogas Production",
            color_continuous_scale="Viridis"
        )
        
        fig_temp_scatter.add_vrect(
            x0=35, x1=40, 
            annotation_text="Optimal Zone", 
            annotation_position="top left",
            fillcolor="green", opacity=0.1,
            line_width=0
        )
        
        fig_temp_scatter.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_temp_scatter, use_container_width=True)
    
    with col2:
        fig_temp_hist = px.histogram(
            df_filtered, 
            x="Temperature (°C)", 
            nbins=20,
            title="Temperature Distribution",
            color_discrete_sequence=['#32CD32']
        )
        
        fig_temp_hist.add_vrect(
            x0=35, x1=40,
            annotation_text="Optimal",
            fillcolor="green", opacity=0.2,
            line_width=0
        )
        
        fig_temp_hist.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_temp_hist, use_container_width=True)
    
    # Feed Loading Analysis
    st.markdown('<div class="section-header"> Feed Loading Analysis (OLR)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
        <h4>🥄 Understanding your Organic Loading Rate (OLR):</h4>
        <p><strong>What it is:</strong> How much organic matter you feed your biogas system daily</p>
        <p><strong>Sweet spot:</strong> Usually 2-4 kg VS/m³/day for optimal production</p>
        <p><strong>Too high:</strong> System gets "indigestion" - pH drops, gas production falls</p>
        <p><strong>Too low:</strong> Bacteria go hungry - less gas production</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig_olr = px.scatter(
        df_filtered, 
        x="OLR (kg VS/m³/day)", 
        y="Biogas Production (m³)",
        color="Temperature (°C)",
        size="pH",
        hover_data=["Timestamp"],
        title="Feed Loading vs Biogas Production",
        color_continuous_scale="RdYlGn"
    )
    
    fig_olr.add_vrect(
        x0=2, x1=4,
        annotation_text="Optimal OLR Zone",
        annotation_position="top left",
        fillcolor="blue", opacity=0.1,
        line_width=0
    )
    
    fig_olr.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_olr, use_container_width=True)
    
    # pH Analysis
    st.markdown('<div class="section-header">⚗️ pH Balance Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚗️ pH: The health indicator of your biogas system:</h4>
        <p><strong>Optimal range:</strong> 6.8 - 7.2 (slightly alkaline)</p>
        <p><strong>Below 6.5:</strong> System is too acidic - bacteria struggle</p>
        <p><strong>Above 7.5:</strong> Too alkaline - also reduces efficiency</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ph_time = px.line(
            df_filtered, 
            x="Timestamp", 
            y="pH",
            title="pH Over Time",
            color_discrete_sequence=['#FF6B6B']
        )
        
        fig_ph_time.add_hline(y=6.8, line_dash="dash", line_color="green", annotation_text="Min Optimal")
        fig_ph_time.add_hline(y=7.2, line_dash="dash", line_color="green", annotation_text="Max Optimal")
        
        fig_ph_time.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ph_time, use_container_width=True)
    
    with col2:
        fig_ph_prod = px.scatter(
            df_filtered, 
            x="pH", 
            y="Biogas Production (m³)",
            color="Temperature (°C)",
            title="pH vs Biogas Production",
            color_continuous_scale="Plasma"
        )
        
        fig_ph_prod.add_vrect(
            x0=6.8, x1=7.2,
            annotation_text="Optimal pH",
            fillcolor="green", opacity=0.1,
            line_width=0
        )
        
        fig_ph_prod.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ph_prod, use_container_width=True)
    
    # Correlation Analysis
    st.markdown('<div class="section-header"> Factor Relationships</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🔗 Understanding the correlation matrix:</h4>
        <p><strong>Dark green:</strong> Strong positive relationship</p>
        <p><strong>Dark red:</strong> Strong negative relationship</p>
        <p><strong>Light colors:</strong> Weak relationship</p>
    </div>
    """, unsafe_allow_html=True)
    
    cols = ['OLR (kg VS/m³/day)', 'Temperature (°C)', 'pH', 'Retention Time (days)', 
            'Moisture Content', 'VS Content', 'Biogas Production (m³)']
    corr = df_filtered[cols].corr()
    
    fig_corr = px.imshow(
        corr,
        color_continuous_scale='RdYlGn',
        aspect="auto",
        title="How Different Factors Relate to Each Other",
        labels=dict(color="Correlation Strength")
    )
    fig_corr.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)

# PRODUCTION FORECAST PAGE
elif app_mode == "prediction":
    st.markdown("""
    <div class="main-header">
        <h1>✨ Production Forecast</h1>
        <p>AI-powered predictions to help you plan ahead</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📅 Select Prediction Date</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pred_date = st.date_input(
            "Choose a future date to predict:", 
            value=date.today(),
            min_value=datetime.today(),
            help="Select any date in the future to get a production forecast"
        )
        st.info(f"🎯 **Prediction Target:** {pred_date.strftime('%B %d, %Y (%A)')}")
    
    with col2:
        if st.button("✨ Generate Prediction", use_container_width=True, type="primary"):
            if 'current_prediction' in st.session_state:
                del st.session_state.current_prediction
            
            with st.spinner("🤖 AI is analyzing your farm patterns..."):
                result = fetch_prediction(str(pred_date))
                st.session_state.current_prediction = result
                st.session_state.prediction_date = pred_date
    
    # Display prediction results
    if 'current_prediction' in st.session_state and st.session_state.current_prediction:
        result = st.session_state.current_prediction
        display_date = st.session_state.prediction_date
        
        if 'predicted_biogas_production' in result:
            predicted_value = result['predicted_biogas_production']
            confidence = result.get('confidence_score', 0.85)
            
            st.markdown('<div class="section-header">✨ AI Prediction Results</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div class="farmer-metric status-excellent" style="text-align: center; padding: 3rem;">
                    <div class="metric-value">{10000*predicted_value:.2f} m³</div>
                    <div class="metric-label">Predicted Production</div>
                    <div class="metric-unit">for {display_date.strftime('%B %d, %Y')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Economic Impact
            st.markdown('<div class="section-header"> Economic Impact</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                revenue_est = predicted_value * 0.65
                st.markdown(f"""
                <div class="farmer-metric status-good">
                    <div class="metric-value">{10000*revenue_est:.2f}Mts</div>
                    <div class="metric-label">Estimated Savings</div>
                    <div class="metric-unit">for this day</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                monthly_est = predicted_value * 30
                st.markdown(f"""
                <div class="farmer-metric status-good">
                    <div class="metric-value">{10000*monthly_est:.2f} m³</div>
                    <div class="metric-label">Monthly Potential</div>
                    <div class="metric-unit">if maintained</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                monthly_revenue = monthly_est * 0.65
                st.markdown(f"""
                <div class="farmer-metric status-good">
                    <div class="metric-value">{10000*monthly_revenue:.2f}Mts</div>
                    <div class="metric-label">Monthly Savings</div>
                    <div class="metric-unit">potential</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.markdown('<div class="section-header"> Prediction Confidence</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                confidence_pct = confidence * 100
                confidence_rating = "Very High" if confidence_pct >= 90 else "High" if confidence_pct >= 80 else "Medium" if confidence_pct >= 70 else "Low"
                st.markdown(f"""
                <div class="info-box">
                    <h4>🤖 AI Prediction Analysis</h4>
                    <p><strong>Confidence Score:</strong> {confidence_pct:.1f}% ({confidence_rating})</p>
                    <p><strong>Prediction Date:</strong> {display_date.strftime('%B %d, %Y (%A)')}</p>
                    <p><strong>What this means:</strong></p>
                    <ul>
                        <li><strong>90%+ confidence:</strong> Very reliable prediction</li>
                        <li><strong>80-90% confidence:</strong> Good prediction, small variations possible</li>
                        <li><strong>70-80% confidence:</strong> Reasonable prediction, monitor conditions</li>
                        <li><strong>Below 70%:</strong> Less certain, consider multiple scenarios</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig_confidence = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence_pct,
                    title={'text': "AI Confidence (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#32CD32"},
                        'steps': [
                            {'range': [0, 70], 'color': "#FFE4E1"},
                            {'range': [70, 85], 'color': "#FFFACD"},
                            {'range': [85, 100], 'color': "#F0FFF0"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))
                fig_confidence.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Recommendations
            st.markdown('<div class="section-header"> Recommendations to Optimize Production</div>', unsafe_allow_html=True)
            
            if predicted_value < 25:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Production seems low. Here's how to improve:</h4>
                    <ul>
                        <li><strong>Check temperature:</strong> Ensure digester stays between 35-40°C</li>
                        <li><strong>Monitor pH:</strong> Keep it between 6.8-7.2</li>
                        <li><strong>Review feeding:</strong> Consistent daily feeding is crucial</li>
                        <li><strong>Check for leaks:</strong> Gas leaks reduce measured production</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif predicted_value > 40:
                st.markdown("""
                <div class="tip-box">
                    <h4>🎉 Excellent production predicted! Keep it up:</h4>
                    <ul>
                        <li><strong>Maintain current practices:</strong> Your system is well-optimized</li>
                        <li><strong>Consider expansion:</strong> High production indicates room for scaling</li>
                        <li><strong>Document what's working:</strong> Record successful practices</li>
                        <li><strong>Plan for excess gas:</strong> Consider additional storage or uses</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <h4>✅ Good production expected. Small optimizations possible:</h4>
                    <ul>
                        <li><strong>Fine-tune temperature:</strong> Move closer to 37°C optimal</li>
                        <li><strong>Optimize feeding schedule:</strong> Consider smaller, more frequent feeds</li>
                        <li><strong>Monitor trends:</strong> Look for gradual improvements</li>
                        <li><strong>Preventive maintenance:</strong> Regular system checks prevent issues</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 How to Use the Production Forecast</h4>
            <ol>
                <li><strong>Select a future date</strong> using the date picker above</li>
                <li><strong>Click "Generate Prediction"</strong> to run our AI analysis</li>
                <li><strong>Review the results</strong> including confidence score and recommendations</li>
                <li><strong>Plan accordingly</strong> based on the predicted production levels</li>
            </ol>
            <p><strong>Best used for:</strong> Planning gas usage, scheduling maintenance, optimizing feeding schedules</p>
        </div>
        """, unsafe_allow_html=True)

# FARM INSIGHTS PAGE
elif app_mode == "insights":
    st.markdown("""
    <div class="main-header">
        <h1> Farm Insights & Tips</h1>
        <p>Practical advice to maximize your biogas production and save money</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get current data for insights
    data = fetch_data(str(date_input))
    
    if data:
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        avg_production = df['Biogas Production (m³)'].mean()
        avg_temp = df['Temperature (°C)'].mean()
        avg_ph = df['pH'].mean()
        production_std = df['Biogas Production (m³)'].std()
        
        st.markdown('<div class="section-header"> Your Current Performance Analysis</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="farmer-metric status-good">
                <div class="metric-value">{10000*avg_production:.2f}</div>
                <div class="metric-label">Current Average</div>
                <div class="metric-unit">m³/day</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            stability = "High" if production_std < 5 else "Medium" if production_std < 10 else "Low"
            status_class = "status-excellent" if stability == "High" else "status-good" if stability == "Medium" else "status-warning"
            st.markdown(f"""
            <div class="farmer-metric {status_class}">
                <div class="metric-value">{stability}</div>
                <div class="metric-label">Production Stability</div>
                <div class="metric-unit">consistency rating</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            efficiency = min(100, (avg_production / 50) * 100)
            st.markdown(f"""
            <div class="farmer-metric status-good">
                <div class="metric-value">{efficiency:.1f}%</div>
                <div class="metric-label">System Efficiency</div>
                <div class="metric-unit">of potential</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Personalized insights
        st.markdown('<div class="section-header"> Personalized Recommendations</div>', unsafe_allow_html=True)
        
        # Temperature insights
        if avg_temp < 30:
            st.markdown(f"""
            <div class="warning-box">
                <h4>🥶 Temperature Too Low - Action Needed!</h4>
                <p><strong>Current average:</strong> {avg_temp:.1f}°C (Optimal: 35-40°C)</p>
                <h5>Immediate actions:</h5>
                <ul>
                    <li><strong>Add insulation:</strong> Wrap digester with blankets or straw</li>
                    <li><strong>Solar heating:</strong> Paint digester black, add transparent cover</li>
                    <li><strong>Compost heating:</strong> Add fresh manure around digester</li>
                    <li><strong>Hot water:</strong> Recirculate hot water through coils</li>
                </ul>
                <p><strong>Expected benefit:</strong> 30-50% increase in gas production</p>
            </div>
            """, unsafe_allow_html=True)
        elif avg_temp > 45:
            st.markdown(f"""
            <div class="warning-box">
                <h4>🔥 Temperature Too High - Cooling Needed!</h4>
                <p><strong>Current average:</strong> {avg_temp:.1f}°C (Optimal: 35-40°C)</p>
                <h5>Cooling strategies:</h5>
                <ul>
                    <li><strong>Add shade:</strong> Cover digester with shade cloth</li>
                    <li><strong>Water cooling:</strong> Spray water on digester during hot days</li>
                    <li><strong>Ventilation:</strong> Improve air circulation around digester</li>
                    <li><strong>Reduce feeding:</strong> Temporarily reduce organic loading</li>
                </ul>
                <p><strong>Risk:</strong> High temperatures can kill beneficial bacteria</p>
            </div>
            """, unsafe_allow_html=True)
        elif 35 <= avg_temp <= 40:
            st.markdown(f"""
            <div class="tip-box">
                <h4>🌡️ Perfect Temperature Control!</h4>
                <p><strong>Current average:</strong> {avg_temp:.1f}°C (Optimal: 35-40°C)</p>
                <p><strong>Keep it up!</strong> Your temperature management is excellent. This optimal range maximizes bacterial activity and gas production.</p>
                <h5>Maintenance tips:</h5>
                <ul>
                    <li><strong>Monitor daily:</strong> Temperature can change with weather</li>
                    <li><strong>Seasonal adjustments:</strong> Prepare for winter insulation needs</li>
                    <li><strong>Document what works:</strong> Record your successful temperature practices</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
                <h4>🌡️ Temperature Needs Minor Adjustment</h4>
                <p><strong>Current average:</strong> {avg_temp:.1f}°C (Optimal: 35-40°C)</p>
                <p>You're close to optimal. Small adjustments could boost production by 10-20%.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # pH insights
        if avg_ph < 6.5:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚗️ pH Too Acidic - System Stress!</h4>
                <p><strong>Current average:</strong> {avg_ph:.1f} (Optimal: 6.8-7.2)</p>
                <h5>Immediate fixes:</h5>
                <ul>
                    <li><strong>Add lime:</strong> Mix 1-2 kg lime into feeding mixture</li>
                    <li><strong>Wood ash:</strong> Add small amounts of wood ash weekly</li>
                    <li><strong>Reduce feeding:</strong> Temporarily cut organic loading by 30%</li>
                    <li><strong>Add buffer:</strong> Mix in crushed eggshells or limestone</li>
                </ul>
                <p><strong>Why it matters:</strong> Acidic conditions kill gas-producing bacteria</p>
            </div>
            """, unsafe_allow_html=True)
        elif avg_ph > 7.5:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚗️ pH Too Alkaline - Needs Balancing!</h4>
                <p><strong>Current average:</strong> {avg_ph:.1f} (Optimal: 6.8-7.2)</p>
                <h5>Balancing actions:</h5>
                <ul>
                    <li><strong>Add organic acids:</strong> Include more kitchen scraps</li>
                    <li><strong>Vinegar treatment:</strong> Small amounts of diluted vinegar</li>
                    <li><strong>Carbon sources:</strong> Add straw, paper, or sawdust</li>
                    <li><strong>Monitor closely:</strong> Test pH every 2-3 days</li>
                </ul>
                <p><strong>Caution:</strong> Make small adjustments and monitor response</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="tip-box">
                <h4>⚗️ pH Balance is Good!</h4>
                <p><strong>Current average:</strong> {avg_ph:.1f} (Optimal: 6.8-7.2)</p>
                <p>Your pH management is working well. Consistent pH in this range supports healthy bacterial communities.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # General optimization tips
    st.markdown('<div class="section-header"> General Optimization Tips</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tip-box">
            <h4>🥄 Feeding Best Practices</h4>
            <ul>
                <li><strong>Consistent timing:</strong> Feed at the same time daily</li>
                <li><strong>Pre-mix ingredients:</strong> Blend different materials before adding</li>
                <li><strong>Optimal ratio:</strong> 1 part fresh manure : 1 part water</li>
                <li><strong>Avoid overfeeding:</strong> Too much food causes acid buildup</li>
                <li><strong>Size matters:</strong> Chop materials into small pieces</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>🔧 Maintenance Schedule</h4>
            <ul>
                <li><strong>Daily:</strong> Check gas pressure and temperature</li>
                <li><strong>Weekly:</strong> Test pH and inspect for leaks</li>
                <li><strong>Monthly:</strong> Remove excess sludge</li>
                <li><strong>Seasonally:</strong> Deep clean and system check</li>
                <li><strong>Annually:</strong> Replace worn parts and seals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tip-box">
            <h4>💰 Maximize Your Savings</h4>
            <ul>
                <li><strong>Peak production:</strong> Use gas during highest production hours</li>
                <li><strong>Storage:</strong> Install gas storage tanks for consistent supply</li>
                <li><strong>Multiple uses:</strong> Cooking, heating, lighting applications</li>
                <li><strong>Slurry benefits:</strong> Use liquid fertilizer for crops</li>
                <li><strong>Carbon credits:</strong> Explore carbon offset opportunities</li>
            </ul>
        </div>
        
        <div class="warning-box">
            <h4>⚠️ Safety Reminders</h4>
            <ul>
                <li><strong>Ventilation:</strong> Ensure proper air flow around equipment</li>
                <li><strong>Leak detection:</strong> Use soap solution to check connections</li>
                <li><strong>Fire safety:</strong> Keep ignition sources away from gas lines</li>
                <li><strong>Regular inspection:</strong> Check all fittings and seals monthly</li>
                <li><strong>Emergency plan:</strong> Know how to quickly shut off gas supply</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Seasonal tips
    st.markdown('<div class="section-header"> Seasonal Management Tips</div>', unsafe_allow_html=True)
    
    current_month = datetime.now().month
    
    if current_month in [12, 1, 2]:  # Summer
        season_tips = """
        <div class="tip-box">
            <h4>☀️ Summer Management (Dec-Feb)</h4>
            <ul>
                <li><strong>Heat control:</strong> Provide shade and cooling</li>
                <li><strong>Water management:</strong> Maintain moisture levels</li>
                <li><strong>Peak production:</strong> Take advantage of warm temperatures</li>
                <li><strong>Ventilation:</strong> Ensure good air circulation</li>
                <li><strong>Monitor closely:</strong> High activity requires more attention</li>
            </ul>
        </div>
        """
    elif current_month in [3, 4, 5]:  # Autumn
        season_tips = """
        <div class="tip-box">
            <h4>🍂 Autumn Management (Mar-May)</h4>
            <ul>
                <li><strong>Prepare for winter:</strong> Start adding insulation</li>
                <li><strong>Harvest season:</strong> Use crop residues as feedstock</li>
                <li><strong>Stable conditions:</strong> Enjoy consistent production</li>
                <li><strong>System check:</strong> Inspect before winter</li>
                <li><strong>Stock supplies:</strong> Prepare winter feeding materials</li>
            </ul>
        </div>
        """
    elif current_month in [6, 7, 8]:  # Winter
        season_tips = """
        <div class="warning-box">
            <h4>❄️ Winter Management (Jun-Aug)</h4>
            <ul>
                <li><strong>Insulation critical:</strong> Wrap digester to retain heat</li>
                <li><strong>Feeding adjustments:</strong> May need to reduce or modify feed</li>
                <li><strong>Lower production:</strong> Expect 20-40% reduction</li>
                <li><strong>Heat sources:</strong> Consider external heating if needed</li>
                <li><strong>Monitor temperature:</strong> Check daily for drops</li>
            </ul>
        </div>
        """
    else:  # Spring
        season_tips = """
        <div class="tip-box">
            <h4>🌱 Spring Management (Sep-Nov)</h4>
            <ul>
                <li><strong>Recovery season:</strong> Production should increase</li>
                <li><strong>Remove winter insulation:</strong> Gradually as temperatures rise</li>
                <li><strong>Fresh start:</strong> Good time for system cleaning</li>
                <li><strong>New materials:</strong> Spring growth provides fresh feedstock</li>
                <li><strong>Optimize settings:</strong> Adjust for improving conditions</li>
            </ul>
        </div>
        """
    
    st.markdown(season_tips, unsafe_allow_html=True)


# SYSTEM DIAGRAMS PAGE
elif app_mode == "diagrams":
    st.markdown("""
    <div class="main-header">
        <h1> Biodigester Technical Drawings</h1>
        <p>Complete architectural plans and component details for this biodigester</p>
    </div>
    """, unsafe_allow_html=True)
    
    # View selector
    st.markdown('<div class="section-header">🔍 Select View Type</div>', unsafe_allow_html=True)
    
    view_tabs = st.tabs(["Side View", "Top View", "Cross Section", "3D Perspective", "Component Details"])
    
    # Side View
    with view_tabs[0]:
        st.markdown("### Side Elevation View - Fixed Dome Biodigester")
        
        # Create side view technical drawing
        fig_side = go.Figure()
        
        # Ground level
        fig_side.add_shape(type="line", x0=0, y0=0, x1=20, y1=0, 
                          line=dict(color="brown", width=3))
        fig_side.add_annotation(x=10, y=-0.5, text=" ", showarrow=False)
        
        # Main digester chamber (underground)
        fig_side.add_shape(type="rect", x0=3, y0=-4, x1=17, y1=0,
                          fillcolor="lightblue", opacity=0.3,
                          line=dict(color="black", width=2))
        fig_side.add_annotation(x=10, y=-2, text="B<br>", 
                               showarrow=False, font=dict(size=10))
        
        # Dome (above ground)
        fig_side.add_shape(type="path", 
                          path="M 3 0 Q 10 3 17 0",
                          fillcolor="lightgreen", opacity=0.3,
                          line=dict(color="black", width=2))
        fig_side.add_annotation(x=10, y=1.5, text="C", 
                               showarrow=False, font=dict(size=10))
        
        # Inlet pipe
        fig_side.add_shape(type="line", x0=0, y0=1, x1=3, y1=-1,
                          line=dict(color="red", width=4))
        fig_side.add_annotation(x=1, y=0, text="A", 
                               showarrow=True, arrowcolor="red")
        
        # Outlet chamber
        fig_side.add_shape(type="rect", x0=17, y0=-2, x1=19, y1=0,
                          fillcolor="orange", opacity=0.3,
                          line=dict(color="black", width=2))
        fig_side.add_annotation(x=18, y=-1, text="F", 
                               showarrow=False, font=dict(size=9))
        
        # Gas pipe
        fig_side.add_shape(type="line", x0=10, y0=2.5, x1=10, y1=4,
                          line=dict(color="green", width=4))
        fig_side.add_annotation(x=10, y=4.5, text="D", 
                               showarrow=True, arrowcolor="green")
        
        # Manhole cover
        fig_side.add_shape(type="rect", x0=9, y0=2.5, x1=11, y1=3,
                          fillcolor="gray", line=dict(color="black", width=1))
        fig_side.add_annotation(x=10, y=2.75, text="E", showarrow=False, font=dict(size=8))
        
        # Dimensions
        fig_side.add_annotation(x=10, y=-5, text="Total Length: 14m", showarrow=False, 
                               font=dict(size=12, color="blue"))
        fig_side.add_annotation(x=-1, y=-2, text="4m", showarrow=False, 
                               font=dict(size=10, color="blue"))
        
        fig_side.update_layout(
            title="Side View - Fixed Dome Biodigester (Scale 1:100)",
            xaxis=dict(range=[-2, 22], showgrid=True, title="Length (meters)"),
            yaxis=dict(range=[-6, 6], showgrid=True, title="Height (meters)"),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_side, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Component Labels:**
            - **A**: Inlet pipe (slurry input)
            - **B**: Main digester chamber
            - **C**: Fixed dome (gas collector)
            - **D**: Gas outlet pipe
            - **E**: Manhole access
            - **F**: Outlet chamber
            """)
        
        with col2:
            st.markdown("""
            **Key Dimensions:**
            - Total length: 14 meters
            - Digester depth: 4 meters
            - Dome height: 3 meters
            - Chamber diameter: 3.2 meters
            - Wall thickness: 150mm
            """)
    
    # Top View
    with view_tabs[1]:
        st.markdown("### Top Plan View - Layout and Components")
        
        fig_top = go.Figure()
        
        # Main circular chamber
        fig_top.add_shape(type="circle", x0=2, y0=2, x1=8, y1=8,
                         fillcolor="lightblue", opacity=0.3,
                         line=dict(color="black", width=2))
        fig_top.add_annotation(x=5, y=5, text="Main<br>Digester", 
                              showarrow=False, font=dict(size=12))
        
        # Inlet chamber (rectangular)
        fig_top.add_shape(type="rect", x0=0, y0=4, x1=2, y1=6,
                         fillcolor="red", opacity=0.3,
                         line=dict(color="black", width=2))
        fig_top.add_annotation(x=1, y=5, text="Inlet<br>Chamber", 
                              showarrow=False, font=dict(size=10))
        
        # Outlet chamber
        fig_top.add_shape(type="rect", x0=8, y0=4, x1=10, y1=6,
                         fillcolor="orange", opacity=0.3,
                         line=dict(color="black", width=2))
        fig_top.add_annotation(x=9, y=5, text="Outlet<br>Chamber", 
                              showarrow=False, font=dict(size=10))
        
        # # Gas pipe connection
        # fig_top.add_shape(type="circle", x0=4.5, y0=4.5, x1=5.5, y1=5.5,
        #                  fillcolor="green", opacity=0.5,
        #                  line=dict(color="black", width=1))
        # fig_top.add_annotation(x=5, y=5, text="Gas<br>Outlet", 
        #                       showarrow=False, font=dict(size=8))
        
        # # Manhole
        # fig_top.add_shape(type="rect", x0=4.2, y0=1.5, x1=5.8, y1=2.5,
        #                  fillcolor="gray", opacity=0.7,
        #                  line=dict(color="black", width=1))
        # fig_top.add_annotation(x=5, y=2, text="Manhole", 
        #                       showarrow=False, font=dict(size=8))
        
        # Connection pipes
        fig_top.add_shape(type="line", x0=2, y0=5, x1=2, y1=5,
                         line=dict(color="red", width=3))
        fig_top.add_shape(type="line", x0=8, y0=5, x1=8, y1=5,
                         line=dict(color="orange", width=3))
        
        # Dimensions and grid
        fig_top.update_layout(
            title="Top Plan View - Biodigester Layout (Scale 1:50)",
            xaxis=dict(range=[-1, 11], showgrid=True, title="Width (meters)"),
            yaxis=dict(range=[0, 10], showgrid=True, title="Length (meters)"),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_top, use_container_width=True)
        
        st.markdown("""
        **Layout Specifications:**
        - Main chamber: Circular, 3.2m diameter
        - Inlet chamber: 2m x 2m rectangular
        - Outlet chamber: 2m x 2m rectangular  
        - Manhole: 60cm x 40cm rectangular access
        - Gas outlet: Central position, 25mm diameter
        - Total footprint: 10m x 8m area required
        """)
    
    # Cross Section
    with view_tabs[2]:
        st.markdown("### Cross-Sectional View - Internal Structure")
        
        fig_cross = go.Figure()
        
        # Outer walls
        fig_cross.add_shape(type="line", x0=1, y0=-4, x1=1, y1=3,
                           line=dict(color="black", width=4))
        fig_cross.add_shape(type="line", x0=9, y0=-4, x1=9, y1=3,
                           line=dict(color="black", width=4))
        
        # Bottom slab
        fig_cross.add_shape(type="line", x0=1, y0=-4, x1=9, y1=-4,
                           line=dict(color="black", width=4))
        
        # Dome structure
        fig_cross.add_shape(type="path", path="M 1 0 Q 5 3 9 0",
                           line=dict(color="black", width=4))
        
        # Slurry level
        fig_cross.add_shape(type="rect", x0=1, y0=-4, x1=9, y1=-1,
                           fillcolor="brown", opacity=0.4)
        fig_cross.add_annotation(x=5, y=-2.5, text="Slurry Level<br>(Active Volume)", 
                                showarrow=False, font=dict(color="white", size=10))
        
        # Gas space
        fig_cross.add_shape(type="path", path="M 1 -1 L 9 -1 Q 5 2 1 -1",
                           fillcolor="yellow", opacity=0.3)
        fig_cross.add_annotation(x=5, y=0.5, text="Gas Collection Space", 
                                showarrow=False, font=dict(size=10))
        
        # Inlet pipe detailed
        fig_cross.add_shape(type="rect", x0=0.5, y0=-1.2, x1=1.5, y1=-0.8,
                           fillcolor="red", opacity=0.6)
        fig_cross.add_annotation(x=1, y=-1, text="Inlet", showarrow=True, 
                                arrowcolor="red", ax=0, ay=-2)
        
        # Outlet pipe
        fig_cross.add_shape(type="rect", x0=8.5, y0=-1.2, x1=9.5, y1=-0.8,
                           fillcolor="orange", opacity=0.6)
        fig_cross.add_annotation(x=9, y=-1, text="Outlet", showarrow=True, 
                                arrowcolor="orange", ax=10, ay=-2)
        
        # Gas outlet
        fig_cross.add_shape(type="circle", x0=4.8, y0=1.8, x1=5.2, y1=2.2,
                           fillcolor="green", opacity=0.7)
        fig_cross.add_annotation(x=5, y=2, text="Manhole", showarrow=True, 
                                arrowcolor="green", ax=5, ay=3)
        
        # Measurements
        fig_cross.add_annotation(x=-0.5, y=-2, text="4m", showarrow=False, 
                                font=dict(size=12, color="blue"))
        fig_cross.add_annotation(x=5, y=-4.5, text="8m diameter", showarrow=False, 
                                font=dict(size=12, color="blue"))
        
        fig_cross.update_layout(
            title="Cross-Sectional View - Internal Structure",
            xaxis=dict(range=[-1, 11], showgrid=True, title="Width (meters)"),
            yaxis=dict(range=[-5, 4], showgrid=True, title="Height (meters)"),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_cross, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Structural Details:**
            - Wall thickness: 150mm reinforced concrete
            - Bottom slab: 200mm thick with waterproofing
            - Dome thickness: 100mm at crown, 150mm at base
            - Reinforcement: Steel bars Ø12mm @ 150mm c/c
            """)
        
        with col2:
            st.markdown("""
            **Operating Levels:**
            - Total depth: 4 meters
            - Slurry level: 3 meters (75% fill)
            - Gas space: 1 meter (25% volume)
            - Freeboard: 0.5 meters safety margin
            """)
    

    import plotly.graph_objects as go
    
    # Function to generate a cylinder
    def create_cylinder(radius, height, center=(0,0,0), zdir="z", color="blue", opacity=0.5):
        theta = np.linspace(0, 2*np.pi, 40)
        z = np.linspace(0, height, 20)
        theta, z = np.meshgrid(theta, z)
        x = radius*np.cos(theta)
        y = radius*np.sin(theta)
        if zdir == "z":
            z = z
        elif zdir == "y":
            y, z = z, y
        elif zdir == "x":
            x, z = z, x
        return go.Surface(x=x+center[0], y=y+center[1], z=z+center[2],
                          colorscale=[[0, color],[1, color]], opacity=opacity, showscale=False)
    
    # Function to generate a spherical dome (spherical cap)
    def create_dome(radius, height, center=(0,0,0), color="green", opacity=0.5):
        phi = np.linspace(0, np.pi/2, 30)   # half sphere (cap)
        theta = np.linspace(0, 2*np.pi, 40)
        phi, theta = np.meshgrid(phi, theta)
        x = radius*np.cos(theta)*np.sin(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(phi)
        # shift dome to sit on top of chamber
        return go.Surface(x=x+center[0], y=y+center[1], z=z+center[2],
                          colorscale=[[0, color],[1, color]], opacity=opacity, showscale=False)
    
    # Main figure
    fig = go.Figure()
    
    # Main chamber (cylinder) underground
    fig.add_trace(create_cylinder(radius=4, height=3.0, center=(0,0,-2), color="lightblue", opacity=0.7))
    
    # Dome gas holder on top
    fig.add_trace(create_dome(radius=4, height=0.8, center=(0,0,0), color="lightgreen", opacity=0.6))
    
    # Inlet pipe (small horizontal cylinder)
    fig.add_trace(create_cylinder(radius=0.2, height=2.5, center=(-6,0,1), zdir="x", color="red", opacity=0.8))
    
    # Outlet pipe (opposite side)
    fig.add_trace(create_cylinder(radius=0.2, height=2.5, center=(3,0,-1), zdir="x", color="orange", opacity=0.8))
    
    # Gas pipe (vertical cylinder)
    fig.add_trace(create_cylinder(radius=0.1, height=3.5, center=(0,0,1.5), zdir="z", color="darkgreen", opacity=1.0))
    
    # Ground plane
    xg = np.linspace(-6, 6, 3)
    yg = np.linspace(-6, 6, 3)
    xg, yg = np.meshgrid(xg, yg)
    zg = np.zeros_like(xg)
    fig.add_trace(go.Surface(x=xg, y=yg, z=zg, colorscale=[[0,"#E0E0E0"],[1,"#E0E0E0"]],
                             opacity=0.3, showscale=False))
    
    # Layout adjustments
    fig.update_layout(
        title="3D Perspective View - Fixed Dome Biodigester",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### System Overview
    - **Main Structure**: Fixed dome biodigester (cylindrical underground chamber + spherical gas dome)  
    - **Capacity**: ~15 m³ (12 m³ digester + 3 m³ gas storage dome)  
    - **Inlet**: Slurry pipe enters chamber at one side  
    - **Outlet**: Effluent outlet at opposite side  
    - **Gas Pipe**: Vertical pipe leaving top of dome  
    - **Construction**: Reinforced concrete with waterproof lining  
    """)

    
    # Component Details
    with view_tabs[4]:
        st.markdown("### Detailed Component Specifications")
        
        components = {
            "Main Digester Chamber": {
                "specs": ["Volume: 15 m³", "Diameter: 3.2 m", "Depth: 4 m", "Wall thickness: 150mm"],
                "materials": ["M25 grade concrete", "TMT steel bars", "Waterproof membrane", "Insulation layer"],
                "color": "#32CD32"
            },
            "Gas Collection Dome": {
                "specs": ["Gas volume: 3 m³", "Height: 1.5 m", "Thickness: 100-150mm", "Pressure: 8-25 cm WC"],
                "materials": ["Reinforced concrete", "Gas-tight membrane", "Pressure relief valve", "Condensate drain"],
                "color": "#4169E1"
            },
            "Inlet System": {
                "specs": ["Pipe diameter: 150mm", "Length: 3 m", "Slope: 1:20", "Daily capacity: 50-75 kg"],
                "materials": ["PVC/concrete pipe", "Inspection chamber", "Gate valve", "Mixing chamber"],
                "color": "#FF6B6B"
            },
            "Outlet System": {
                "specs": ["Chamber size: 1m x 1m x 1m", "Overflow level control", "Sludge removal access"],
                "materials": ["Concrete chamber", "Outlet pipe 100mm", "Level indicators", "Access cover"],
                "color": "#FFB84D"
            },
            "Gas Pipeline": {
                "specs": ["Main line: 25mm GI pipe", "Pressure: 10-20 cm WC", "Flow rate: 2-4 m³/day"],
                "materials": ["Galvanized iron pipes", "Ball valves", "Pressure gauge", "Water trap"],
                "color": "#9B59B6"
            }
        }
        
        for component, details in components.items():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div style="background: {details['color']}; padding: 1.5rem; border-radius: 10px; 
                           color: white; text-align: center; margin: 1rem 0;">
                    <h4>{component}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                Technical Specifications:
                {chr(10).join(f'• {spec}' for spec in details['specs'])}
                
                Materials & Components:
                {chr(10).join(f'• {material}' for material in details['materials'])}
                """)
        
        # Construction sequence
        st.markdown("""
        ### Construction Sequence
        
        Phase 1: Site Preparation (Week 1)
        1. Site survey and soil testing
        2. Excavation to required depth (4.5m including foundation)
        3. Foundation concrete pouring (M20 grade, 200mm thick)
        4. Waterproofing membrane installation
        
        Phase 2: Structure Construction (Weeks 2-3)
        5. Wall construction with reinforcement (150mm thick)
        6. Dome formwork and reinforcement setup
        7. Dome concrete pouring (M25 grade)
        8. Curing period (7-14 days with water sprinkling)
        
        Phase 3: Installation (Week 4)
        9. Inlet and outlet pipe connections
        10. Gas pipeline installation with valves
        11. Manhole cover and access ladder installation
        12. Pressure testing and leak detection
        
        Phase 4: Commissioning (Weeks 5-6)
        13. Initial slurry feeding (50% capacity)
        14. Bacterial culture inoculation
        15. System testing and parameter adjustment
        16. Training and handover to user
        """)
        
        # Material quantities
        st.markdown("""
        ### Material Quantities (15 m³ System)
        
        | Material | Quantity | Unit | Purpose |
        |----------|----------|------|---------|
        | Concrete M25 | 8.5 | m³ | Main structure |
        | Concrete M20 | 2.0 | m³ | Foundation |
        | TMT Steel 12mm | 450 | kg | Reinforcement |
        | TMT Steel 8mm | 120 | kg | Stirrups |
        | Waterproof membrane | 85 | m² | Waterproofing |
        | PVC pipes 150mm | 6 | m | Inlet system |
        | GI pipes 25mm | 15 | m | Gas line |
        | Gate valves 150mm | 2 | nos | Inlet control |
        | Ball valves 25mm | 3 | nos | Gas control |
        | Pressure gauge | 1 | nos | Monitoring |
        | Manhole cover | 1 | nos | Access |
        """)
    
    
    # Troubleshooting diagram
    st.markdown('<div class="section-header"> Common Problems & Solutions</div>', unsafe_allow_html=True)
    
    problems_solutions = {
        "Low Gas Production": {
            "symptoms": ["Less than expected output", "Weak flame", "Gas runs out quickly"],
            "causes": ["Low temperature", "Wrong pH", "Insufficient feeding", "System leaks"],
            "solutions": ["Add insulation", "Test and adjust pH", "Increase feeding rate", "Check all connections"],
            "color": "#FF6B6B"
        },
        "No Gas Production": {
            "symptoms": ["No gas flow", "Empty gas holder", "Burner won't light"],
            "causes": ["System too cold", "pH too acidic", "Blocked pipes", "New system (needs time)"],
            "solutions": ["Heat system", "Add lime/ash", "Clear blockages", "Wait 2-4 weeks"],
            "color": "#DC143C"
        },
        "Bad Smell": {
            "symptoms": ["Strong sulfur smell", "Rotten egg odor", "Smell around system"],
            "causes": ["High sulfur content", "Imbalanced feeding", "Poor ventilation"],
            "solutions": ["Add iron filings", "Balance carbon/nitrogen", "Improve ventilation"],
            "color": "#FF8C00"
        },
        "Unstable Production": {
            "symptoms": ["Production varies daily", "Sometimes high, sometimes low"],
            "causes": ["Inconsistent feeding", "Temperature fluctuations", "pH swings"],
            "solutions": ["Regular feeding schedule", "Better insulation", "Monitor pH closely"],
            "color": "#FFD700"
        }
    }
    
    for problem, details in problems_solutions.items():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="color: {details['color']};">⚠️ {problem}</h4>
                <h5>Symptoms:</h5>
                <ul>
                    {''.join(f'<li>{symptom}</li>' for symptom in details['symptoms'])}
                </ul>
                <h5>Likely Causes:</h5>
                <ul>
                    {''.join(f'<li>{cause}</li>' for cause in details['causes'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="tip-box">
                <h4 style="color: {details['color']};">💡 Solutions</h4>
                <ol>
                    {''.join(f'<li><strong>{solution.split()[0]}:</strong> {" ".join(solution.split()[1:])}</li>' for solution in details['solutions'])}
                </ol>
                <p><strong>Prevention:</strong> Regular monitoring and maintenance</p>
            </div>
            """, unsafe_allow_html=True)


# SYSTEM DIAGRAMS PAGE
elif app_mode == "diagrams":
    st.markdown("""
    <div class="main-header">
        <h1>📐 System Diagrams & Setup Guide</h1>
        <p>Visual guides to understand and optimize your biogas system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Biogas process diagram
    st.markdown('<div class="section-header">🔄 Biogas Production Process</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Understanding the Biogas Production Process</h4>
        <p>Biogas production happens in four main stages inside your digester:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create process flow diagram
        fig_process = go.Figure()
        
        # Add process steps
        steps = ['Organic Matter\n(Manure, Kitchen Scraps)', 'Hydrolysis\n(Breaking Down)', 
                'Acidification\n(Acid Production)', 'Methanogenesis\n(Gas Production)']
        x_pos = [1, 2, 3, 4]
        
        for i, (step, x) in enumerate(zip(steps, x_pos)):
            fig_process.add_trace(go.Scatter(
                x=[x], y=[1], 
                mode='markers+text',
                marker=dict(size=100, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#32CD32'][i]),
                text=step,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                showlegend=False
            ))
        
        # Add arrows
        for i in range(len(x_pos)-1):
            fig_process.add_annotation(
                x=x_pos[i]+0.4, y=1,
                ax=x_pos[i+1]-0.4, ay=1,
                arrowhead=2, arrowsize=2, arrowwidth=3,
                arrowcolor="gray"
            )
        
        fig_process.update_layout(
            title="Biogas Production Process Flow",
            xaxis=dict(range=[0.5, 4.5], showticklabels=False, showgrid=False),
            yaxis=dict(range=[0.5, 1.5], showticklabels=False, showgrid=False),
            height=300,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_process, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="tip-box">
            <h5>🔬 Process Details:</h5>
            <p><strong>Stage 1:</strong> Complex organic matter breaks into simple compounds</p>
            <p><strong>Stage 2:</strong> Acid-producing bacteria create organic acids</p>
            <p><strong>Stage 3:</strong> Methane-producing bacteria convert acids to biogas</p>
            <p><strong>Result:</strong> 50-70% Methane, 30-40% CO₂, traces of H₂S</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System components diagram
    st.markdown('<div class="section-header">🏗️ Biogas System Components</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-container">
            <h4>🏗️ Main System Components</h4>
            <div style="text-align: center; padding: 2rem;">
                <div style="background: #32CD32; padding: 1rem; margin: 0.5rem; border-radius: 10px; color: white;">
                    <strong>1. Digester Tank</strong><br>
                    Where fermentation happens<br>
                    <small>Size: 8-20 m³ for family farm</small>
                </div>
                <div style="background: #4169E1; padding: 1rem; margin: 0.5rem; border-radius: 10px; color: white;">
                    <strong>2. Gas Holder</strong><br>
                    Collects and stores biogas<br>
                    <small>Pressure regulation system</small>
                </div>
                <div style="background: #FF6B6B; padding: 1rem; margin: 0.5rem; border-radius: 10px; color: white;">
                    <strong>3. Inlet Pipe</strong><br>
                    Feeds organic matter in<br>
                    <small>Daily feeding point</small>
                </div>
                <div style="background: #FFB84D; padding: 1rem; margin: 0.5rem; border-radius: 10px; color: white;">
                    <strong>4. Outlet Chamber</strong><br>
                    Removes spent slurry<br>
                    <small>Rich liquid fertilizer</small>
                </div>
                <div style="background: #9B59B6; padding: 1rem; margin: 0.5rem; border-radius: 10px; color: white;">
                    <strong>5. Gas Pipeline</strong><br>
                    Delivers gas to appliances<br>
                    <small>Includes safety valves</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tip-box">
            <h4>🔧 Installation Tips</h4>
            <ul>
                <li><strong>Location:</strong> Close to livestock for easy feeding</li>
                <li><strong>Sunlight:</strong> South-facing for natural warming</li>
                <li><strong>Drainage:</strong> Slightly sloped ground for outlet</li>
                <li><strong>Access:</strong> Easy to reach for daily maintenance</li>
                <li><strong>Safety:</strong> Away from living areas (20m minimum)</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>📏 Sizing Your System</h4>
            <p><strong>Family of 5:</strong> 6-8 m³ digester</p>
            <p><strong>Small farm (10 cattle):</strong> 15-20 m³</p>
            <p><strong>Medium farm (25 cattle):</strong> 35-50 m³</p>
            <p><strong>Rule of thumb:</strong> 1 m³ per animal unit</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Troubleshooting diagram
    st.markdown('<div class="section-header"> Common Problems & Solutions</div>', unsafe_allow_html=True)
    
    problems_solutions = {
        "Low Gas Production": {
            "symptoms": ["Less than expected output", "Weak flame", "Gas runs out quickly"],
            "causes": ["Low temperature", "Wrong pH", "Insufficient feeding", "System leaks"],
            "solutions": ["Add insulation", "Test and adjust pH", "Increase feeding rate", "Check all connections"],
            "color": "#FF6B6B"
        },
        "No Gas Production": {
            "symptoms": ["No gas flow", "Empty gas holder", "Burner won't light"],
            "causes": ["System too cold", "pH too acidic", "Blocked pipes", "New system (needs time)"],
            "solutions": ["Heat system", "Add lime/ash", "Clear blockages", "Wait 2-4 weeks"],
            "color": "#DC143C"
        },
        "Bad Smell": {
            "symptoms": ["Strong sulfur smell", "Rotten egg odor", "Smell around system"],
            "causes": ["High sulfur content", "Imbalanced feeding", "Poor ventilation"],
            "solutions": ["Add iron filings", "Balance carbon/nitrogen", "Improve ventilation"],
            "color": "#FF8C00"
        },
        "Unstable Production": {
            "symptoms": ["Production varies daily", "Sometimes high, sometimes low"],
            "causes": ["Inconsistent feeding", "Temperature fluctuations", "pH swings"],
            "solutions": ["Regular feeding schedule", "Better insulation", "Monitor pH closely"],
            "color": "#FFD700"
        }
    }
    
    for problem, details in problems_solutions.items():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="color: {details['color']};">⚠️ {problem}</h4>
                <h5>Symptoms:</h5>
                <ul>
                    {''.join(f'<li>{symptom}</li>' for symptom in details['symptoms'])}
                </ul>
                <h5>Likely Causes:</h5>
                <ul>
                    {''.join(f'<li>{cause}</li>' for cause in details['causes'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="tip-box">
                <h4 style="color: {details['color']};">💡 Solutions</h4>
                <ol>
                    {''.join(f'<li><strong>{solution.split()[0]}:</strong> {" ".join(solution.split()[1:])}</li>' for solution in details['solutions'])}
                </ol>
                <p><strong>Prevention:</strong> Regular monitoring and maintenance</p>
            </div>
            """, unsafe_allow_html=True)

# SETTINGS PAGE
elif app_mode == "settings":
    st.markdown("""
    <div class="main-header">
        <h1> Settings & Configuration</h1>
        <p>Customize your dashboard and manage your account</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"> Account Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>👤 User Profile</h4>
            <p><strong>Username:</strong> """ + st.session_state.username + """</p>
            <p><strong>Account Type:</strong> Farm Manager</p>
            <p><strong>System Access:</strong> Full Dashboard</p>
            <p><strong>Last Login:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tip-box">
            <h4>🔐 Security Status</h4>
            <p><strong>Login Status:</strong> ✅ Secure</p>
            <p><strong>Session:</strong> Active</p>
            <p><strong>Data Protection:</strong> Enabled</p>
            <p><strong>Backup Status:</strong> Up to date</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dashboard preferences
    st.markdown('<div class="section-header"> Dashboard Preferences</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Display Settings")
        
        refresh_rate = st.selectbox(
            "Data Refresh Rate",
            options=["Real-time", "Every 5 minutes", "Every 15 minutes", "Every 30 minutes", "Manual only"],
            index=2,
            help="How often the dashboard updates with new data"
        )
        
        units = st.selectbox( 
            "Units System",
            options=["Metric (m³, °C, kg)", "Imperial (ft³, °F, lbs)", "Mixed"],
            index=0,
            help="Choose your preferred unit system"
        )
        
        theme = st.selectbox(
            "Dashboard Theme",
            options=["Green (Default)"], #"Dark Mode" , "Blue", "High Contrast"
            index=0,
            help="Visual theme for the dashboard"
        )
        
        default_date_range = st.selectbox(
            "Default Date Range",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Current month", "Custom"],
            index=1,
            help="Default time period for data analysis"
        )
    
    with col2:
        st.markdown("### Notification Settings")
        
        enable_alerts = st.checkbox("Enable System Alerts", value=True, help="Get notified of important system events")
        
        if enable_alerts:
            st.write("Alert Types:")
            low_production_alert = st.checkbox("Low Production Warning", value=True)
            temperature_alert = st.checkbox("Temperature Out of Range", value=True)
            ph_alert = st.checkbox("pH Imbalance Alert", value=True)
            maintenance_reminder = st.checkbox("Maintenance Reminders", value=True)
            
            alert_method = st.selectbox(
                "Alert Method",
                options=["Dashboard Only", "Email", "SMS", "WhatsApp", "All Methods"],
                index=0,
                help="How you want to receive alerts"
            )
        
        st.markdown("### AI Assistant Settings")
        
        ai_enabled = st.checkbox("Enable AI Assistant (BioAIgent)", value=True, help="Turn on/off the AI chat feature")
        
        if ai_enabled:
            ai_detail_level = st.selectbox(
                "AI Response Detail",
                options=["Concise", "Detailed", "Technical", "Beginner-friendly"],
                index=1,
                help="How detailed AI responses should be"
            )
            
            ai_language = st.selectbox(
                "AI Language",
                options=["English", "Português", "Español", "Français"],
                index=0,
                help="Language for AI responses"
            )
    
    # System configuration
    st.markdown('<div class="section-header"> System Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Sources")
        
        api_endpoint = st.text_input(
            "API Endpoint",
            value=BASE_URL,
            help="Backend API server address"
        )
        
        data_collection_interval = st.selectbox(
            "Data Collection Frequency",
            options=["Every hour", "Every 6 hours", "Daily", "Manual"],
            index=0,
            help="How often sensors collect data"
        )
        
        backup_enabled = st.checkbox("Enable Data Backup", value=True, help="Automatically backup your data")
        
        if backup_enabled:
            backup_frequency = st.selectbox(
                "Backup Frequency",
                options=["Daily", "Weekly", "Monthly"],
                index=0
            )
    
    with col2:
        st.markdown("### System Limits")
        
        max_temp_alert = st.number_input(
            "Maximum Temperature Alert (°C)",
            min_value=30.0,
            max_value=60.0,
            value=45.0,
            step=0.5,
            help="Temperature threshold for alerts"
        )
        
        min_temp_alert = st.number_input(
            "Minimum Temperature Alert (°C)",
            min_value=0.0,
            max_value=35.0,
            value=25.0,
            step=0.5,
            help="Low temperature threshold for alerts"
        )
        
        min_production_alert = st.number_input(
            "Low Production Alert (m³/day)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Production level that triggers alerts"
        )
        
        max_ph = st.number_input("Max pH Alert", min_value=6.0, max_value=9.0, value=7.5, step=0.1)
        min_ph = st.number_input("Min pH Alert", min_value=5.0, max_value=7.0, value=6.5, step=0.1)
    
    # Action buttons
    st.markdown('<div class="section-header"> Save & Actions</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Settings", use_container_width=True, type="primary"):
            st.success("✅ Settings saved successfully!")
            st.balloons()
    
    with col2:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.warning("⚠️ All settings will be reset to default values.")
    
    with col3:
        if st.button("📤 Export Data", use_container_width=True):
            st.info("📊 Data export feature will be available soon.")
    
    with col4:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.success("👋 Logged out successfully!")
            time.sleep(1)
            st.rerun()
    
    # System information
    st.markdown('<div class="section-header">ℹ️ System Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 System Status</h4>
            <p><strong>Dashboard Version:</strong> 2.1.0</p>
            <p><strong>Last Update:</strong> 2025-03-15</p>
            <p><strong>Database Status:</strong> ✅ Connected</p>
            <p><strong>AI Service:</strong> ✅ Active</p>
            <p><strong>Prediction Engine:</strong> ✅ Running</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tip-box">
            <h4>📞 Support & Help</h4>
            <p><strong>User Guide:</strong> <a href="#" style="color: #32CD32;">Download PDF</a></p>
            <p><strong>Video Tutorials:</strong> <a href="#" style="color: #32CD32;">Watch Online</a></p>
            <p><strong>Technical Support:</strong> support@gpf-biogas.com</p>
            <p><strong>Emergency:</strong> +27-XXX-XXXX</p>
            <p><strong>Community Forum:</strong> <a href="#" style="color: #32CD32;">Join Discussion</a></p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer">
    <h3>Green Poutry Farm Mozambique</h3>
    <p>Empowering farmers with inteligent biogas technology</p>
    <p>© 2025 Copyright All Right Reserved.</p>
    <p><small>Dashboard Version 2.1.0 | Last Updated: September 2025</small></p>
</div>
""", unsafe_allow_html=True)



                    