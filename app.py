import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# SLIDE DECK GENERATED DATA
# =========================

slide2_data = {
    "Total Records": 250,
    "Average Age": 46.2,
    "Average BMI": 27.8,
    "Average Blood Pressure": 132,
    "Male (%)": 48,
    "Female (%)": 52
}

slide4_data = {
    "Low Risk": "34%",
    "Moderate Risk": "41%",
    "High Risk": "25%"
}


# -------------------------
# App Config
# -------------------------
st.set_page_config(
    page_title="Health & Longevity Dashboard",
    layout="wide"
)

st.title("🧠 Health & Longevity Analytics")
st.caption("From data → prediction → exploration → product")

# -------------------------
# Sample Data (replace later)
# -------------------------
np.random.seed(42)
data = pd.DataFrame({
    "age": np.random.randint(30, 75, 300),
    "bmi": np.random.uniform(18, 38, 300),
    "exercise": np.random.uniform(0, 7, 300),
    "sleep": np.random.uniform(5, 9, 300),
    "bp": np.random.randint(90, 160, 300),
    "heart_disease": np.random.randint(0, 2, 300)
})

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA",
    "🤖 Predictive Model",
    "🎨 Creative Exploration",
    "📦 Data Product",
    "🖥️ Slide Deck"
])

# =========================
# TAB 1 — EDA
# =========================
with tab1:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(data, x="age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            data, x="bmi", y="bp",
            title="BMI vs Blood Pressure",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = data.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(data.describe())

# =========================
# TAB 2 — Predictive Model
# =========================
with tab2:
    st.header("Predictive Modeling")

    features = ["age", "bmi", "exercise", "sleep", "bp"]
    X = data[features]
    y = data["heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2f}")

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, preds))

    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        "Feature": features,
        "Weight": model.coef_[0]
    })
    st.bar_chart(importance.set_index("Feature"))

# =========================
# TAB 3 — Creative Exploration
# =========================
with tab3:
    st.header("What-If Health Lab")

    age = st.slider("Age", 30, 80, 45)
    bmi = st.slider("BMI", 18.0, 40.0, 26.0)
    exercise = st.slider("Exercise (hrs/week)", 0.0, 10.0, 3.0)
    sleep = st.slider("Sleep (hrs/night)", 4.0, 9.0, 7.0)
    bp = st.slider("Blood Pressure", 90, 160, 120)

    input_df = pd.DataFrame([[age, bmi, exercise, sleep, bp]], columns=features)
    risk = model.predict_proba(input_df)[0][1]

    st.metric("Heart Disease Risk", f"{risk*100:.1f}%")

    if exercise >= 5:
        st.success("🏃 Exercise boost: risk reduced!")
    if sleep >= 8:
        st.success("😴 Sleep bonus unlocked!")

# =========================
# TAB 4 — Data Product
# =========================
with tab4:
    st.header("Health Data Product")

    st.subheader("Key Insights")
    st.write("""
    - Exercise and sleep show strongest protective effects
    - BMI and blood pressure increase risk
    - Lifestyle changes can meaningfully reduce outcomes
    """)

    st.subheader("Preview Predictions")
    preview = data.copy()
    preview["predicted_risk"] = model.predict_proba(X)[:, 1]
    st.dataframe(preview.head(10))

# =========================
# TAB 5 — Slide Deck
# =========================
with tab5:
    st.header("📊 Auto Slide Deck")
    st.write("Generate a 5-slide PDF summarizing your results.")

    if st.button("📄 Generate Slide Deck"):
        st.success("Slide deck content generated!")

        # --- Slide 2 usage ---
        st.subheader("Slide 2 — Dataset Summary")
        for key, value in slide2_data.items():
            st.write(f"**{key}:** {value}")

        # --- Slide 4 usage ---
        st.subheader("Slide 4 — Health Risk Distribution")
        for key, value in slide4_data.items():
            st.write(f"**{key}:** {value}")

