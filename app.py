import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

forecast_model=joblib.load('models/energy_forecast_model.pkl')
anomaly_model=joblib.load('models/anomaly_detector.pkl')

def generate_recommendations(df):
    recommendations = []
    for _, row in df.iterrows():
        tips = []
        if row.get('is_weekend', 0) == 1 and row.get('global_active_power', 0) > 3.5:
            tips.append("📈 High weekend usage – reduce AC/TV usage.")
        if row.get('voltage', 240) < 230:
            tips.append("⚡ Low voltage – check appliances.")
        if row.get('anomaly', 1) == -1:
            tips.append("🚨 Spike detected – check idle/faulty devices.")
        if row.get('global_active_power', 0) > 4.5:
            tips.append("💡 High usage – switch to energy-efficient devices.")
        recommendations.append(" | ".join(tips) if tips else "✅ All good!")
    df['recommendations'] = recommendations
    return df