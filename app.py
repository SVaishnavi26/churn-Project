import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_files():
    model = joblib.load(os.path.join(BASE, "model", "churn_model.joblib"))
    scaler = joblib.load(os.path.join(BASE, "model", "scaler.joblib"))
    columns = joblib.load(os.path.join(BASE, "model", "columns.joblib"))
    threshold = joblib.load(os.path.join(BASE, "model", "threshold.joblib"))
    return model, scaler, columns, threshold


model, scaler, columns, threshold = load_files()

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details to see how likely they are to leave. "
    "Built with logistic regression on the Telco Customer Churn dataset."
)

# ---------- Inputs ----------
st.subheader("Account")
c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months with the company)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0, step=1.0)
with c2:
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless = st.checkbox("Paperless billing", value=True)

st.subheader("Services")
c3, c4 = st.columns(2)
with c3:
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    phone = st.checkbox("Phone service", value=True)
with c4:
    multiple_lines = st.checkbox("Multiple lines", value=False)

has_internet = internet != "No"
if has_internet:
    st.caption("Internet add-ons")
    a1, a2, a3 = st.columns(3)
    with a1:
        online_security = st.checkbox("Online security")
        online_backup = st.checkbox("Online backup")
    with a2:
        device_protection = st.checkbox("Device protection")
        tech_support = st.checkbox("Tech support")
    with a3:
        streaming_tv = st.checkbox("Streaming TV")
        streaming_movies = st.checkbox("Streaming movies")
else:
    online_security = online_backup = device_protection = False
    tech_support = streaming_tv = streaming_movies = False

st.subheader("Customer")
c5, c6 = st.columns(2)
with c5:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.checkbox("Senior citizen")
with c6:
    partner = st.checkbox("Has a partner")
    dependents = st.checkbox("Has dependents")

# ---------- Build the row exactly like the training data ----------
# "No internet service" / "No phone service" were merged into "No" in training
if not phone:
    multiple_lines = False

total_charges = tenure * monthly  # estimate: total is roughly tenure x monthly

row = {col: 0 for col in columns}
row.update({
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": int(senior),
    "Partner": int(partner),
    "Dependents": int(dependents),
    "tenure": tenure,
    "PhoneService": int(phone),
    "MultipleLines": int(multiple_lines),
    "OnlineSecurity": int(online_security),
    "OnlineBackup": int(online_backup),
    "DeviceProtection": int(device_protection),
    "TechSupport": int(tech_support),
    "StreamingTV": int(streaming_tv),
    "StreamingMovies": int(streaming_movies),
    "PaperlessBilling": int(paperless),
    "MonthlyCharges": monthly,
    "TotalCharges": total_charges,
    "InternetService_Fiber optic": int(internet == "Fiber optic"),
    "InternetService_No": int(internet == "No"),
    "Contract_One year": int(contract == "One year"),
    "Contract_Two year": int(contract == "Two year"),
    "PaymentMethod_Credit card (automatic)": int(payment == "Credit card (automatic)"),
    "PaymentMethod_Electronic check": int(payment == "Electronic check"),
    "PaymentMethod_Mailed check": int(payment == "Mailed check"),
})

X_new = pd.DataFrame([row])[columns]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X_new[num_cols] = scaler.transform(X_new[num_cols])

# ---------- Prediction ----------
st.divider()
if st.button("Predict churn risk", type="primary"):
    prob = float(model.predict_proba(X_new)[0, 1])

    st.metric("Churn probability", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= threshold:
        st.error(
            f"⚠️ Likely to leave (probability is above the {threshold:.0%} threshold). "
            "Consider a retention offer, such as a discount for a longer contract."
        )
    else:
        st.success(
            f"✅ Likely to stay (probability is below the {threshold:.0%} threshold)."
        )

    st.caption(
        "Total charges are estimated as tenure × monthly charges. "
        "This is a learning project, not a production tool."
    )
