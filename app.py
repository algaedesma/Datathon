import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_xgb.pkl")
feature_names = model.get_booster().feature_names

st.title("📱 Prediksi Tingkat Kecanduan Ponsel Remaja")

# Input form
gender = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"])
school = st.selectbox("Jenjang Sekolah", ["SMP", "SMA", "Kuliah"])
daily_usage = st.slider("Jam penggunaan per hari", 0.0, 24.0, 5.0)
checks = st.slider("Frekuensi cek ponsel per hari", 0, 100, 30)
social = st.slider("Waktu media sosial (jam)", 0.0, 10.0, 2.0)
gaming = st.slider("Waktu bermain game (jam)", 0.0, 10.0, 1.0)
edu = st.slider("Waktu untuk edukasi (jam)", 0.0, 10.0, 1.5)
purpose = st.selectbox("Tujuan penggunaan ponsel", ["Communication", "Entertainment", "Education", "Other"])

# Mapping & preprocessing
gender_map = {"Female": 0, "Male": 1, "Other": 2}
grade_map = {"SMP": 0, "SMA": 1, "Kuliah": 2}

data = {
    "Gender": gender_map[gender],
    "School_Grade": grade_map[school],
    "Daily_Usage_Hours": daily_usage,
    "Phone_Checks_Per_Day": checks,
    "Time_on_Social_Media": social,
    "Time_on_Gaming": gaming,
    "Time_on_Education": edu,
    "Phone_Usage_Purpose_Education": int(purpose == "Education"),
    "Phone_Usage_Purpose_Entertainment": int(purpose == "Entertainment"),
    "Phone_Usage_Purpose_Other": int(purpose == "Other"),
    "Total_App_Time": social + gaming + edu,
    "Phone_Checks_Per_Hour": checks / (daily_usage + 0.1)
}

# PREPROCESS
df = pd.DataFrame([data])

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(df)[0]
    st.success(f"Tingkat kecanduan diprediksi: **{prediction:.2f}** dari 10")
