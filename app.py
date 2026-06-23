import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Load Model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1E3A8A;
    font-size: 42px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #6B7280;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<p class="main-title">🎓 Placement Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Predict whether a student is likely to get placed based on CGPA and IQ.</p>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    This Machine Learning model predicts
    placement chances using:

    • CGPA
    • IQ Score
    """
)

# Input Section
st.subheader("📋 Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        step=0.01,
        value=7.0
    )

with col2:
    iq = st.number_input(
        "IQ Score",
        min_value=0,
        max_value=200,
        value=100
    )

st.write("")

# Prediction Button
if st.button("🚀 Predict Placement", use_container_width=True):

    data = np.array([[cgpa, iq]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    # Probability (if model supports it)
    try:
        probability = model.predict_proba(data_scaled)[0]
        placed_prob = probability[1] * 100
    except:
        placed_prob = None

    st.divider()

    if prediction[0] == 1:
        st.success("✅ Student is Likely to be Placed")

        if placed_prob is not None:
            st.metric(
                label="Placement Probability",
                value=f"{placed_prob:.2f}%"
            )

    else:
        st.error("❌ Student is Likely NOT to be Placed")

        if placed_prob is not None:
            st.metric(
                label="Placement Probability",
                value=f"{placed_prob:.2f}%"
            )

# Footer
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")