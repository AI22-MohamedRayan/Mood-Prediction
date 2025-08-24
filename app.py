import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    df = pd.read_csv("steps_tracker_dataset_cleaned.csv")
    X = df.drop(columns=['mood', 'date'])
    y = df['mood']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = DecisionTreeClassifier(criterion='gini', random_state=42)
    model.fit(X, y_encoded)
    return model, le, X.columns  # return column names too

model, le, feature_names = load_model()

st.title("Mood Predictor Based on Daily Activity")
# Sidebar info
with st.sidebar:
    st.title("🧠 About This Project")
    st.markdown("""
    This Streamlit app predicts your **mood** based on your daily activities like:
    - Steps taken
    - Distance covered
    - Calories burned
    - Active minutes
    - Sleep hours
    - Water intake
    
    The model is trained on real-world activity tracking data.
    """)


# Input form
steps = st.number_input("Steps", min_value=0.0)
distance_km = st.number_input("Distance (km)", min_value=0.0)
calories_burned = st.number_input("Calories Burned", min_value=0.0)
active_minutes = st.number_input("Active Minutes", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
water_intake = st.number_input("Water Intake (liters)", min_value=0.0)

if st.button("Predict Mood"):
    # 1) build a DataFrame with the same columns the model expects
    input_df = pd.DataFrame(
        [[steps, distance_km, calories_burned, active_minutes, sleep_hours, water_intake]],
        columns=feature_names
    )
    # 2) predict from the DataFrame
    prediction = model.predict(input_df)
    mood = le.inverse_transform(prediction)[0]
    st.success(f"Your predicted mood is: **{mood}**")
