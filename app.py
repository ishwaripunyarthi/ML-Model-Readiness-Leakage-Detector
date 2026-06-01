import streamlit as st
import pandas as pd
import requests

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ML Model Readiness Checker",
    layout="centered"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main { background-color: transparent; }

h1, h2, h3, h4 {
    color: #ffffff;
}

.card {
    background: rgba(255, 255, 255, 0.12);
    padding: 20px;
    border-radius: 15px;
    margin: 15px 0px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.metric {
    font-size: 22px;
    font-weight: bold;
    color: #00e5ff;
}

.success-box {
    background-color: rgba(0, 255, 127, 0.2);
    padding: 15px;
    border-radius: 12px;
    border-left: 6px solid #00ff7f;
}

.error-box {
    background-color: rgba(255, 99, 71, 0.2);
    padding: 15px;
    border-radius: 12px;
    border-left: 6px solid #ff6347;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🛡️ ML Model Readiness & Leakage Detector</h1>",
    unsafe_allow_html=True
)

# -----------------------------
# REQUIREMENTS CARD
# -----------------------------
st.markdown("""
<div class="card">
<h3>📄 Dataset Upload Requirements</h3>
<ul>
<li>CSV format only</li>
<li>Must contain a time-related column</li>
<li>Target column must be binary (0/1)</li>
<li>Numeric features preferred</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV file",
    type=["csv"]
)

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully")

    # Preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Column selection
    st.subheader("🔧 Column Selection")

    target_col = st.selectbox(
        "Select Target Column",
        df.columns
    )

    time_col = st.selectbox(
        "Select Time Column",
        df.columns
    )

    st.write("🎯 Target selected:", target_col)
    st.write("⏳ Time selected:", time_col)

    # Button
    run_eval = st.button(
        "🚀 Run Model Evaluation"
    )

    # -----------------------------
    # API CALL TO FASTAPI
    # -----------------------------
    if run_eval:

        with st.spinner("Running evaluation..."):

            url = "http://127.0.0.1:8000/predict"

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            data = {
                "target_col": target_col,
                "time_col": time_col
            }

            try:

                response = requests.post(
                    url,
                    files=files,
                    data=data
                )

                if response.status_code == 200:

                    results = response.json()

                else:

                    st.error("API request failed")
                    st.stop()

            except Exception as e:

                st.error("Could not connect to FastAPI server")
                st.write(e)
                st.stop()

        # -----------------------------
        # RESULTS DISPLAY
        # -----------------------------
        st.markdown(
            "<h2>📊 Evaluation Results</h2>",
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="card">
            <p class="metric">
            Random Split AUC: {results['Random Split AUC']:.3f}
            </p>

            <p class="metric">
            Time-Based AUC: {results['Time-Based Split AUC']:.3f}
            </p>

            <p class="metric">
            Leakage Score: {results['Leakage Score']:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<h2>🧠 Model Decision</h2>",
            unsafe_allow_html=True
        )

        # Decision box
        if "SAFE" in results["Deployment Decision"]:

            st.markdown(f"""
            <div class="success-box">
            ✅ <strong>
            {results['Deployment Decision']}
            </strong><br>
            {results['Leakage Verdict']}
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="error-box">
            ❌ <strong>
            {results['Deployment Decision']}
            </strong><br>
            {results['Leakage Verdict']}
            </div>
            """, unsafe_allow_html=True)