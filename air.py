# air_quality_app.py

import streamlit as st
import requests
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---------------------- API KEYS ----------------------
WEATHER_KEY = "7a6bdc6503b6effb94f5273c5017bb79"
AQICN_TOKEN = "57e81cc987e25ab67e2b3593f686565695476486"
CSV_PATH = Path("city_day.csv")
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
TARGET = "AQI_Bucket"
CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# ---------------------- UTILS FOR MODULE 1 ----------------------
def get_user_city():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        return res.get("city", "")
    except:
        return ""

def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") != 200:
            return None
        return {
            "city": res["name"],
            "temp": res["main"]["temp"],
            "weather": res["weather"][0]["description"]
        }
    except:
        return None

def get_aqi(city):
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
        res = requests.get(url).json()
        if res["status"] != "ok":
            return None
        data = res["data"]
        iaqi = data.get("iaqi", {})
        return {
            "aqi": data["aqi"],
            "pm25": iaqi.get("pm25", {}).get("v", "N/A"),
            "pm10": iaqi.get("pm10", {}).get("v", "N/A"),
            "co": iaqi.get("co", {}).get("v", "N/A")
        }
    except:
        return None

def aqi_level(aqi):
    if aqi <= 50: return "Good 🟢"
    if aqi <= 100: return "Moderate 🟡"
    if aqi <= 150: return "Unhealthy for Sensitive 🟠"
    if aqi <= 200: return "Unhealthy 🔴"
    if aqi <= 300: return "Very Unhealthy ⚫"
    return "Hazardous ☠️"

def get_background_color(aqi):
    if aqi <= 50: return "#d4edda"
    elif aqi <= 100: return "#fff3cd"
    elif aqi <= 150: return "#ffeeba"
    elif aqi <= 200: return "#f8d7da"
    elif aqi <= 300: return "#d6d8db"
    return "#f5c6cb"

# ---------------------- ML MODEL (MODULE 2) ----------------------
@st.cache_resource(show_spinner=False)
def load_and_train():
    if not CSV_PATH.exists():
        st.error(f"Dataset '{CSV_PATH.name}' not found. Place it in the same folder.")
        st.stop()
    df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET])
    X = df[FEATURES]
    y = df[TARGET].map({lab: i for i, lab in enumerate(CATEGORY_ORDER)})
    int_to_label = {i: lab for i, lab in enumerate(CATEGORY_ORDER)}

    pipeline = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), FEATURES)
        ])),
        ("rf", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    return pipeline, int_to_label, acc

# ---------------------- STREAMLIT APP ----------------------
st.set_page_config("Air Quality & Weather App", page_icon="🌍", layout="centered")
st.title("🌍 Air Quality Suite")
st.write("Choose a module below to check AQI, weather, or predict category from pollutants.")

tab1, tab2 = st.tabs(["📍 Real-Time AQI & Weather", "🔬 Predict AQI Category (ML)"])

# ---------------------- MODULE 1 ----------------------
with tab1:
    st.header("📍 Real-Time Weather and AQI Checker")
    use_geo = st.checkbox("📌 Detect my location automatically")
    city = get_user_city() if use_geo else st.text_input("Enter City Name")

    if st.button("Check", key="check_weather"):
        if not city:
            st.warning("Enter a city name.")
        else:
            weather = get_weather(city)
            aqi_data = get_aqi(city)

            if weather:
                st.subheader(f"📍 {weather['city']}")
                st.write(f"🌤 **Weather:** {weather['weather'].title()}")
                st.write(f"🌡️ **Temperature:** {weather['temp']} °C")
            else:
                st.error("❌ Weather data not found.")

            if aqi_data:
                bg_color = get_background_color(aqi_data["aqi"])
                st.markdown(
                    f"""<div style="background-color: {bg_color}; padding: 15px; border-radius: 10px;">
                        <h4>🧪 AQI: {aqi_data['aqi']} ({aqi_level(aqi_data['aqi'])})</h4>
                        <p>PM2.5: {aqi_data['pm25']} µg/m³</p>
                        <p>PM10: {aqi_data['pm10']} µg/m³</p>
                        <p>CO: {aqi_data['co']} µg/m³</p>
                    </div>""",
                    unsafe_allow_html=True)
            else:
                st.error("❌ AQI data not available.")

# ---------------------- MODULE 2 ----------------------
with tab2:
    st.header("🔬 Predict AQI Category from Pollutants")
    with st.spinner("Loading model..."):
        model, idx_to_label, val_acc = load_and_train()

    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.number_input("PM2.5", 0.0, 1000.0, step=1.0)
        no2 = st.number_input("NO2", 0.0, 1000.0, step=1.0)
        co = st.number_input("CO", 0.0, 1000.0, step=1.0)
    with col2:
        pm10 = st.number_input("PM10", 0.0, 1000.0, step=1.0)
        so2 = st.number_input("SO2", 0.0, 1000.0, step=1.0)
        o3 = st.number_input("O3", 0.0, 1000.0, step=1.0)
    with col3:
        nh3 = st.number_input("NH3", 0.0, 1000.0, step=1.0)

    features_df = pd.DataFrame([[pm25, pm10, no2, so2, co, o3, nh3]], columns=FEATURES)

    if st.button("🔮 Predict AQI Category", key="predict_aqi"):
        if features_df.isna().any().any():
            st.warning("Fill in all pollutant values.")
        else:
            pred = int(model.predict(features_df)[0])
            label = idx_to_label[pred]
            st.success(f"### Predicted Category: {label}")

            proba = model.predict_proba(features_df)[0]
            st.dataframe(pd.DataFrame({
                "Category": [idx_to_label[i] for i in range(len(proba))],
                "Probability": proba
            }).sort_values("Probability", ascending=False), use_container_width=True)
