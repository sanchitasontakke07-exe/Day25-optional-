import streamlit as st
import pandas as pd
import joblib

# Load Files
model = joblib.load("student_model.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Student Result Prediction")

st.title("🎓 Student Result Prediction System")

st.write("Enter Student Details")

data = {}

for col in columns:

    if col in encoders:

        options = list(encoders[col].classes_)

        value = st.selectbox(col, options)

        value = encoders[col].transform([value])[0]

        data[col] = value

    else:

        value = st.number_input(col, value=0)

        data[col] = value

input_df = pd.DataFrame([data])

if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    result = encoders["Result"].inverse_transform([prediction])[0]

    if result == "Pass":
        st.success("Prediction : PASS ✅")
    else:
        st.error("Prediction : FAIL ❌")

