import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Vehicle Insurance Fraud Prediction",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state["page"] = "splash"

# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("insurance_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

# =========================================================
# SPLASH SCREEN
# =========================================================
if st.session_state["page"] == "splash":

    st.markdown("""
        <style>
        .center-container {
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .glow {
            font-size: 64px;
            font-weight: bold;
            color: #00f7ff;
            animation: glow 1.5s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #00f7ff; }
            to { text-shadow: 0 0 35px #00f7ff; }
        }
        </style>

        <div class="center-container">
            <div class="glow">VEHICLE INSURANCE</div>
            <div>Fraud Prediction System</div>
        </div>
    """, unsafe_allow_html=True)

    time.sleep(3)
    st.session_state["page"] = "app"
    st.rerun()

# =========================================================
# MAIN APPLICATION
# =========================================================
elif st.session_state["page"] == "app":

    model, scaler, feature_columns = load_artifacts()

    st.title("Vehicle Accident Insurance Fraud Prediction")
    st.write("AI-based evaluation using **Machine Learning + Real Insurance Rules**")

    st.subheader("📝 Enter Claim Details")

    # ---------------- BASIC DETAILS ----------------
    vehicle_age = st.number_input("Vehicle Age (Years)", 0, 50, 5)
    driver_age = st.number_input("Driver Age", 18, 80, 30)
    past_claims = st.number_input("Previous Claims Count", 0, 10, 0)

    # ---------------- INSURANCE DETAILS ----------------
    insurance_type_ui = st.selectbox(
        "Insurance Type",
        ["Third Party", "Comprehensive", "No Insurance"]
    )

    accident_type_ui = st.selectbox(
        "Accident Type",
        ["Minor", "Major", "Fire", "Theft"]
    )

    accident_area_ui = st.selectbox("Accident Area", ["Urban", "Rural"])
    drunk_ui = st.selectbox("Drunk Driving?", ["No", "Yes"])

    # ---------------- ADVANCED DETAILS ----------------
    claim_amount = st.number_input("Claim Amount (₹)", 1000, 5000000, 50000)
    vehicle_value = st.number_input("Vehicle Market Value (₹)", 10000, 5000000, 300000)

    days_since_policy = st.number_input(
        "Days Between Policy Start & Accident",
        0, 3650, 180
    )

    police_report_ui = st.selectbox("Police / Fire Report Filed?", ["Yes", "No"])

    fire_cause_ui = st.selectbox(
        "Fire Cause (if applicable)",
        ["Not Applicable", "Technical Fault", "Short Circuit", "Owner Negligence"]
    )

    # ---------------- ENCODING ----------------
    insurance_map = {"Third Party": 0, "Comprehensive": 1, "No Insurance": -1}
    accident_map = {"Minor": 0, "Major": 1, "Fire": 2, "Theft": 3}
    fire_cause_map = {
        "Not Applicable": 0,
        "Technical Fault": 1,
        "Short Circuit": 2,
        "Owner Negligence": 3
    }

    insurance_type = insurance_map[insurance_type_ui]
    accident_type = accident_map[accident_type_ui]
    fire_cause = fire_cause_map[fire_cause_ui]
    accident_area = 1 if accident_area_ui == "Urban" else 0
    drunk = 1 if drunk_ui == "Yes" else 0
    police_report = 1 if police_report_ui == "Yes" else 0

    # ---------------- FEATURE TEMPLATE ----------------
    input_df = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    feature_mapping = {
        "AgeOfVehicle": vehicle_age,
        "Age": driver_age,
        "PastNumberOfClaims": past_claims,
        "PolicyType": insurance_type,
        "AccidentType": accident_type,
        "AccidentArea": accident_area,
        "DrunkDriving": drunk
    }

    for col, val in feature_mapping.items():
        if col in input_df.columns:
            input_df[col] = val

    # =========================================================
    # BUSINESS RULES (FINAL – EXACTLY AS YOU WANTED)
    # =========================================================
    rejection_reasons = []
    approval_reasons = []
    risk_score = 0

    # NO INSURANCE
    if insurance_type == -1:
        rejection_reasons.append(
            "Vehicle was not insured at the time of accident – claim is legally invalid"
        )
        risk_score += 40

    # THEFT
    if accident_type == 3:
        if police_report == 0:
            rejection_reasons.append(
                "Theft reported without FIR / police complaint – mandatory document missing"
            )
            risk_score += 40
        elif insurance_type != 1:
            rejection_reasons.append(
                "Theft claims are not covered under third-party insurance policy"
            )
            risk_score += 30
        else:
            approval_reasons.append(
                "Theft verified with FIR and covered under comprehensive insurance policy"
            )

    # FIRE
    if accident_type == 2:

        if fire_cause == 3:
            rejection_reasons.append(
                "Fire caused due to owner negligence – insurance policy void even with FIR"
            )
            risk_score += 45

        elif fire_cause in [1, 2]:
            if insurance_type != 1:
                rejection_reasons.append(
                    "Fire damage is not covered under third-party insurance policy"
                )
                risk_score += 30
            else:
                approval_reasons.append(
                    "Fire caused by technical fault and verified by authorities – eligible claim"
                )

        elif police_report == 0:
            rejection_reasons.append(
                "Fire incident without police or fire department verification"
            )
            risk_score += 30

    # DRUNK DRIVING
    if drunk == 1:
        rejection_reasons.append(
            "Drunk driving detected – violation of motor vehicle insurance terms"
        )
        risk_score += 25

    # MULTIPLE CLAIMS
    if past_claims >= 3:
        rejection_reasons.append(
            "Multiple past insurance claims indicate suspicious claim behaviour"
        )
        risk_score += 15

    # CLAIM AMOUNT
    if claim_amount > vehicle_value * 0.8:
        rejection_reasons.append(
            "Claim amount is disproportionately high compared to vehicle market value"
        )
        risk_score += 15

    # POLICY TIMING
    if days_since_policy < 30:
        rejection_reasons.append(
            "Accident occurred shortly after policy purchase – high fraud risk"
        )
        risk_score += 10

    risk_score = min(risk_score, 100)

    # ---------------- ML PREDICTION ----------------
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # ---------------- FINAL DECISION ----------------
    if st.button("Predict Claim"):

        st.subheader("Insurance Claim Evaluation Report")
        st.metric("Fraud Risk Score", f"{risk_score} / 100")

        if prediction == 1 or risk_score >= 50 or rejection_reasons:

            st.error(" CLAIM REJECTED – POLICY & FRAUD VIOLATION")

            st.markdown("### Reasons for Rejection:")
            for r in rejection_reasons:
                st.markdown(f"• **{r}**")

            st.markdown("""
            ### 🛑 Final Verdict
            Based on **AI fraud detection**, **policy validation**,  
            and **legal compliance checks**, this claim is classified  
            as **INVALID** and **will not be processed further**.
            """)

        else:
            st.success("CLAIM APPROVED – ELIGIBLE FOR SETTLEMENT")

            st.markdown("### Approval Reasons:")
            for r in approval_reasons:
                st.markdown(f"• **{r}**")

            st.markdown("""
            ### 📌 Next Steps
            1. Document verification  
            2. Vehicle inspection  
            3. Claim amount assessment  
            4. Settlement initiation  

            👉 Claim approved for insurance processing.
            """)
