import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🎓 Placement Prediction System")

cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0)

iq = st.number_input("Enter IQ Score", min_value=0)

if st.button("Predict"):

    data = np.array([[cgpa, iq]])
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Student is likely to be Placed")
    else:
        st.error("❌ Student is likely NOT to be Placed")