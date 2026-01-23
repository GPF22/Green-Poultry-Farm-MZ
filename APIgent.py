from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import google.generativeai as genai
import os
from fastapi.middleware.cors import CORSMiddleware

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")  # Set your API key
genai.configure(api_key=GEMINI_API_KEY)

# Load pre-trained model and scaler
model = joblib.load("model_RFR.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Biogas Production Prediction API with AI Chat")

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schemas
class PredictionRequest(BaseModel):
    date: Optional[str] = None  # Format: "YYYY-MM-DD"

class DataRequest(BaseModel):
    date: str = "2025-03-31"

class ChatRequest(BaseModel):
    message: str
    user_data: Optional[Dict[str, Any]] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Biogas Production Prediction API with AI Chat"}

# Endpoint: Retrieve data for last 2 weeks ending on selected date
@app.post("/get_data")
def retrieve_data(request: DataRequest):
    try:
        conn = sqlite3.connect("biogas.db")
        df = pd.read_sql("SELECT * FROM biogas_data", conn, parse_dates=["Timestamp"])
        conn.close()
        
        end_date = pd.to_datetime(request.date)
        start_date = end_date - pd.Timedelta(weeks=2)
        df_filtered = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)]
        
        return df_filtered.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

# Endpoint: Predict for a selected date
@app.post("/get_prediction")
def get_prediction(request: PredictionRequest):
    try:
        selected_date = request.date
        conn = sqlite3.connect("biogas.db")
        df = pd.read_sql("SELECT * FROM biogas_data", conn, parse_dates=["Timestamp"])
        conn.close()
        
        if df.empty:
            return {"error": "No data available in the database."}
        
        if selected_date:
            try:
                target_date = pd.to_datetime(selected_date)
                df["date_diff"] = abs(df["Timestamp"] - target_date)
                closest = df.sort_values("date_diff").iloc[0]
            except Exception as e:
                return {"error": f"Invalid date format or error finding closest record: {str(e)}"}
        else:
            closest = df.sort_values("Timestamp", ascending=False).iloc[0]
        
        # Ensure derived columns
        if "DayOfWeek" not in closest:
            closest["DayOfWeek"] = closest["Timestamp"].dayofweek
        if "HourOfDay" not in closest:
            closest["HourOfDay"] = closest["Timestamp"].hour
        
        features = [
            closest["OLR (kg VS/m³/day)"],
            closest["Temperature (°C)"],
            closest["pH"],
            closest["Retention Time (days)"],
            closest["Moisture Content"],
            closest["VS Content"],
            closest["DayOfWeek"],
            closest["HourOfDay"]
        ]
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)
        
        return {
            "input_date": selected_date if selected_date else str(closest["Timestamp"].date()),
            "used_timestamp": str(closest["Timestamp"]),
            "predicted_biogas_production": round(float(prediction[0]), 6)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# New endpoint: Chat with Gemini AI
@app.post("/chat_gemini")
def chat_with_gemini(request: ChatRequest):
    try:
        # Initialize Gemini model
        model_ai = genai.GenerativeModel('gemini-pro')
        
        # Create context from user data if available
        context = ""
        if request.user_data:
            context = f"""
            Current farm status:
            {chr(10).join([f"- {k}: {v}" for k, v in request.user_data.items()])}
            
            """
        
        # Create a comprehensive biogas expert prompt
        system_prompt = f"""You are an expert biogas production consultant specializing in African farming conditions, particularly in Mozambique. You have deep knowledge about:

        - Anaerobic digestion processes
        - Biogas system optimization
        - Troubleshooting common problems
        - Feed management and organic loading rates
        - Temperature and pH control
        - Safety procedures
        - Cost-effective maintenance
        - Local farming conditions in Southern Africa

        {context}

        Provide practical, actionable advice in a friendly, farmer-focused manner. Use simple language and include specific numbers, ratios, or steps when possible. Always prioritize safety and cost-effectiveness.

        User question: {request.message}
        """
        
        # Generate response
        response = model_ai.generate_content(system_prompt)
        
        return {
            "response": response.text,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "response": f"I'm having trouble connecting right now. Please try again later. Technical details: {str(e)}",
            "status": "error"
        }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)