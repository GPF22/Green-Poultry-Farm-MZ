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

# ── Gemini config ─────────────────────────────────────────────────────────────
# Set your key in the environment — never hardcode it.
# Windows:   set GEMINI_API_KEY=AIza...
# Mac/Linux: export GEMINI_API_KEY=AIza...
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA4vocWethJC8hEws-uCVtCSi3oUTrUh3Q") #AIzaSyA4vocWethJC8hEws-uCVtCSi3oUTrUh3Q
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Load pre-trained model and scaler
model  = joblib.load("model_RFR.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Biogas Production Prediction API with AI Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    date: Optional[str] = None

class DataRequest(BaseModel):
    date: str = "2025-03-31"

class ChatRequest(BaseModel):
    message: str
    user_data: Optional[Dict[str, Any]] = None

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": f"Biogas Prediction API — chat powered by Gemini ({GEMINI_MODEL})"}

# ── Data retrieval ────────────────────────────────────────────────────────────
@app.post("/get_data")
def retrieve_data(request: DataRequest):
    try:
        conn = sqlite3.connect("biogas.db")
        df = pd.read_sql("SELECT * FROM biogas_data", conn, parse_dates=["Timestamp"])
        conn.close()
        end_date   = pd.to_datetime(request.date)
        start_date = end_date - pd.Timedelta(weeks=2)
        df_filtered = df[(df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)]
        return df_filtered.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

# ── Prediction ────────────────────────────────────────────────────────────────
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
            target_date = pd.to_datetime(selected_date)
            df["date_diff"] = abs(df["Timestamp"] - target_date)
            closest = df.sort_values("date_diff").iloc[0]
        else:
            closest = df.sort_values("Timestamp", ascending=False).iloc[0]
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
            closest["HourOfDay"],
        ]
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(features_scaled)
        return {
            "input_date": selected_date if selected_date else str(closest["Timestamp"].date()),
            "used_timestamp": str(closest["Timestamp"]),
            "predicted_biogas_production": round(float(prediction[0]), 6),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ── Gemini chat ───────────────────────────────────────────────────────────────
@app.post("/chat_gemini")   # same endpoint — dashboard needs no changes
def chat_with_gemini(request: ChatRequest):
    if not GEMINI_API_KEY:
        return {
            "response": "Gemini API key not set. Run: set GEMINI_API_KEY=AIza... (Windows) or export GEMINI_API_KEY=AIza... (Mac/Linux), then restart the server.",
            "status": "error",
        }
    try:
        # Build prompt
        if request.user_data and "system_context" in request.user_data:
            full_prompt = request.user_data["system_context"]
        else:
            context = ""
            if request.user_data:
                context = "\n".join([f"- {k}: {v}" for k, v in request.user_data.items()])
            full_prompt = (
                "You are an expert biogas production consultant specialising in African farming "
                "conditions, particularly in Mozambique. You have deep knowledge about anaerobic "
                "digestion, system optimisation, troubleshooting, feed management, temperature and "
                "pH control, safety, and cost-effective maintenance.\n"
                + (f"\nCurrent farm status:\n{context}\n" if context else "")
                + "\nProvide practical, actionable advice in a friendly, farmer-focused manner. "
                "Use simple language and include specific numbers, ratios, or steps when possible. "
                "Always prioritise safety and cost-effectiveness. Be thorough and detailed.\n\n"
                f"User question: {request.message}"
            )

        ai_model = genai.GenerativeModel(GEMINI_MODEL)
        response  = ai_model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7,
            )
        )

        return {
            "response": response.text,
            "status":   "success",
            "model":    GEMINI_MODEL,
        }

    except Exception as e:
        return {"response": f"Gemini error: {str(e)}", "status": "error"}

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "model":     GEMINI_MODEL,
        "key_set":   bool(GEMINI_API_KEY),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
