import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Models
lstm_model = load_model(os.path.join(BASE_DIR, "lstm_model.h5"))
gru_model = load_model(os.path.join(BASE_DIR, "gru_model.h5"))

# Load Scaler
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Title
st.title("Railway Equipment Failure Prediction")

st.write("Enter sensor values")

# Inputs
air_temp = st.number_input("Air Temperature [K]", value=298.0)
process_temp = st.number_input("Process Temperature [K]", value=308.0)
speed = st.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=100.0)

# Prediction Button
if st.button("Predict"):

    # Input array
    input_data = np.array([[air_temp, process_temp, speed, torque, tool_wear]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Create sequence
    sequence = np.repeat(input_scaled, 10, axis=0)
    sequence = sequence.reshape(1, 10, 5)

    # Predictions
    lstm_prob = float(lstm_model.predict(sequence)[0][0])
    gru_prob = float(gru_model.predict(sequence)[0][0])

    avg_prob = (lstm_prob + gru_prob) / 2

    result = "Failure Predicted" if avg_prob >= 0.5 else "No Failure"

    # Output
    st.subheader("Prediction Result")

    st.write("LSTM Probability:", round(lstm_prob * 100, 2), "%")
    st.write("GRU Probability:", round(gru_prob * 100, 2), "%")
    st.write("Average Probability:", round(avg_prob * 100, 2), "%")

    st.success(result)