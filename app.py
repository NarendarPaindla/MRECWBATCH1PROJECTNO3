import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load pretrained models
forecast_model = joblib.load("models/energy_forecast_model.pkl")
anomaly_model = joblib.load("models/anomaly_detector.pkl")

# Generate SmartWatt tips
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

# App layout
st.set_page_config(page_title="SmartWatt Dashboard", layout="wide")
st.title("⚡ SmartWatt – AI-Powered Household Energy Dashboard")

uploaded_file = st.file_uploader("📁 Upload your energy dataset (.csv or .txt)", type=["csv", "txt"])

if uploaded_file:
    with st.spinner("Processing your file..."):
        try:
            # Step 1: Load raw file
            df = pd.read_csv(uploaded_file, sep=';', na_values='?', low_memory=False)
            df.columns = [col.strip().lower() for col in df.columns]

            # Step 2: Rename important columns if needed
            df.rename(columns={
                'global_active_power': 'global_active_power',
                'global_reactive_power': 'global_reactive_power',
                'voltage': 'voltage',
                'global_intensity': 'global_intensity',
                'sub_metering_1': 'sub_metering_1',
                'sub_metering_2': 'sub_metering_2',
                'sub_metering_3': 'sub_metering_3',
                'date': 'date',
                'time': 'time'
            }, inplace=True)

            # Step 3: Merge date + time → datetime
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
            df.drop(columns=['date', 'time'], inplace=True)
            df.dropna(subset=['datetime'], inplace=True)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

            # Step 4: Convert to numeric
            numeric_cols = df.columns
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(inplace=True)

            # Step 5: Resample to daily
            daily_df = df.resample('D').mean()
            daily_df['day_of_week'] = daily_df.index.dayofweek
            daily_df['is_weekend'] = daily_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

            # Step 6: Forecast next day's usage
            required_cols = ['global_active_power', 'voltage', 'is_weekend', 'day_of_week']
            if all(col in daily_df.columns for col in required_cols):
                last_row = daily_df[required_cols].tail(1)
                next_day_prediction = forecast_model.predict(last_row)[0]
            else:
                next_day_prediction = None
                st.warning("⚠️ Some required columns for forecasting are missing!")

            # Step 7: Detect anomalies
            if 'global_active_power' in daily_df.columns:
                daily_df['anomaly'] = anomaly_model.predict(daily_df[['global_active_power']])
                daily_df['anomaly_label'] = daily_df['anomaly'].apply(lambda x: "⚠️ Anomaly" if x == -1 else "✅ Normal")
            else:
                daily_df['anomaly'] = 1
                daily_df['anomaly_label'] = "✅ Normal"

            # Step 8: Generate recommendations
            daily_df = generate_recommendations(daily_df)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.stop()

    # Dashboard display
    st.subheader("📊 Daily Energy Usage (kW)")
    st.line_chart(daily_df['global_active_power'])

    st.subheader("🚨 Detected Anomalies")
    st.dataframe(daily_df[daily_df['anomaly'] == -1][['global_active_power', 'anomaly_label']])

    st.subheader("💡 SmartWatt Recommendations")
    st.dataframe(daily_df[['global_active_power', 'recommendations']].tail(7))

    if next_day_prediction is not None:
        st.success(f"📈 Forecasted energy usage for tomorrow: **{next_day_prediction:.2f} kW**")

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_df.index, daily_df['global_active_power'], label='Usage (kW)', color='blue')
    ax.scatter(daily_df[daily_df['anomaly'] == -1].index,
               daily_df[daily_df['anomaly'] == -1]['global_active_power'],
               color='red', label='Anomaly')
    ax.set_title("Energy Usage with Anomalies")
    ax.set_ylabel("kW")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Upload a `.csv` or `.txt` file with household power data to begin.")
