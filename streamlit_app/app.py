import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telecom Customer Prediction", page_icon="📱", layout="centered")

# Custom CSS for a modern look
st.markdown("""
    <style>
        .main {background-color: #f7f7fa;}
        .stTabs [data-baseweb="tab-list"] {justify-content: center;}
        .stTabs [data-baseweb="tab"] {font-size: 18px;}
        .stButton>button {background-color: #4F8BF9; color: white;}
        .stButton>button:hover {background-color: #1746A2;}
        .stSelectbox>div>div>div>div {font-size: 16px;}
        .stNumberInput>div>div>input {font-size: 16px;}
        .stForm {background: #fff; border-radius: 12px; padding: 2rem; box-shadow: 0 2px 8px #00000010;}
    </style>
""", unsafe_allow_html=True)

st.title("📱 Telecom Customer Prediction")

tab1, tab2 = st.tabs(["🔮 Churn Check", "💸 Monthly Charges Check"])

with tab1:
    st.header("🔮 Churn Prediction")
    with st.form("churn_form"):
        st.markdown("#### Enter Customer Details")
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, value=22)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=89.1)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=1949.4)
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        st.markdown("")
        submit_churn = st.form_submit_button("🚦 Predict Churn")

    if submit_churn:
        with st.spinner("Predicting..."):
            original_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'gender': [gender],
                'Partner': [partner],
                'Dependents': [dependents],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'SeniorCitizen': [senior_citizen]
            })

            onehot = joblib.load("../models/onehot_encoder_gender_partner_dependents.pkl")
            freq_encoders = joblib.load("../models/frequency_encoders.pkl")
            monthly_scaler = joblib.load("../models/monthlycharges_scaler.pkl")
            totalcharges_scaler = joblib.load("../models/totalcharges_minmax_scaler.pkl")
            model = joblib.load("../models/logistic_regression_telco.pkl")

            original_data['MonthlyCharges_scaled'] = monthly_scaler.transform(original_data[['MonthlyCharges']])
            original_data['TotalCharges_scaled'] = totalcharges_scaler.transform(original_data[['TotalCharges']])

            onehot_df = pd.DataFrame(
                onehot.transform(original_data[['gender', 'Partner', 'Dependents']]),
                columns=onehot.get_feature_names_out(['gender', 'Partner', 'Dependents'])
            )
            original_data = pd.concat([original_data, onehot_df], axis=1)
            original_data.drop(['gender', 'Partner', 'Dependents'], axis=1, inplace=True)

            for col in freq_encoders:
                fe_col = col + '_FE'
                original_data[fe_col] = original_data[col].map(freq_encoders[col])
                original_data.drop(col, axis=1, inplace=True)

            model_features = [
                'tenure', 'MonthlyCharges_scaled', 'TotalCharges_scaled', 'gender_Male', 'gender_Female',
                'OnlineBackup_FE', 'InternetService_FE', 'OnlineSecurity_FE', 'TechSupport_FE', 'Contract_FE',
                'PaymentMethod_FE', 'PaperlessBilling_FE', 'MultipleLines_FE', 'DeviceProtection_FE',
                'Partner_Yes', 'Partner_No', 'SeniorCitizen', 'Dependents_No', 'Dependents_Yes',
                'StreamingTV_FE', 'StreamingMovies_FE'
            ]
            X_new = original_data[model_features]

            predictions = model.predict(X_new)
            probabilities = model.predict_proba(X_new)[:, 1]

            st.markdown("---")
            st.subheader("Prediction Result")
            st.metric("Churn Probability", f"{probabilities[0]:.2%}")
            if predictions[0] == 1:
                st.error("🚨 YES, customer churn will happen.")
            else:
                st.success("✅ NO, customer will NOT churn.")

with tab2:
    st.header("💸 Monthly Charges Prediction")
    with st.form("monthlycharges_form"):
        st.markdown("#### Enter Customer Details")
        col1, col2 = st.columns(2)
        with col1:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="mc_internet_service")
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], key="mc_streaming_tv")
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"], key="mc_streaming_movies")
            device_protection = st.selectbox("Device Protection", ["Yes", "No"], key="mc_device_protection")
            online_security = st.selectbox("Online Security", ["Yes", "No"], key="mc_online_security")
            online_backup = st.selectbox("Online Backup", ["Yes", "No"], key="mc_online_backup")
            tech_support = st.selectbox("Tech Support", ["Yes", "No"], key="mc_tech_support")
        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"], key="mc_multiple_lines")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="mc_phone_service")
            tenure = st.number_input("Tenure (months)", min_value=0, value=22, key="mc_tenure")
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="mc_payment_method")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="mc_contract")
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="mc_paperless_billing")
            churn = st.selectbox("Churn", ["No", "Yes"], key="mc_churn")
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], key="mc_senior_citizen")
            partner = st.selectbox("Partner", ["Yes", "No"], key="mc_partner")
        st.markdown("")
        submit_mc = st.form_submit_button("💸 Predict Monthly Charges")

    if submit_mc:
        with st.spinner("Predicting..."):
            original_data = pd.DataFrame({
                'InternetService': [internet_service],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'DeviceProtection': [device_protection],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'TechSupport': [tech_support],
                'MultipleLines': [multiple_lines],
                'PhoneService': [phone_service],
                'tenure': [tenure],
                'PaymentMethod': [payment_method],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'Churn': [churn],
                'SeniorCitizen': [senior_citizen],
                'Partner': [partner]
            })

            # Convert Churn to numeric
            original_data['Churn'] = original_data['Churn'].map({'No': 0, 'Yes': 1})

            freq_encoders = joblib.load("../models/frequency_encoders.pkl")

            for col in freq_encoders:
                fe_col = col + '_FE'
                original_data[fe_col] = original_data[col].map(freq_encoders[col])
                if col not in ['Churn']:
                    original_data.drop(col, axis=1, inplace=True)

            original_data['Partner_Yes'] = [1 if original_data.get('Partner', ['No'])[0] == 'Yes' else 0]
            original_data['Partner_No'] = [1 if original_data.get('Partner', ['No'])[0] == 'No' else 0]
            if 'Partner' in original_data.columns:
                original_data.drop('Partner', axis=1, inplace=True)

            model_features = [
                'InternetService_FE', 'StreamingTV_FE', 'StreamingMovies_FE',
                'DeviceProtection_FE', 'OnlineSecurity_FE', 'OnlineBackup_FE', 'TechSupport_FE',
                'MultipleLines_FE', 'PhoneService_FE', 'tenure', 'PaymentMethod_FE', 'Contract_FE',
                'PaperlessBilling_FE', 'Churn', 'SeniorCitizen', 'Partner_No', 'Partner_Yes'
            ]
            X_new = original_data[model_features]

            model = joblib.load("../models/best_random_forest_monthlycharges.pkl")
            predicted_monthlycharges = model.predict(X_new)[0]

            churn_label = "Yes" if original_data['Churn'].iloc[0] == 1 else "No"
            st.markdown("---")
            st.subheader("Prediction Result")
            st.metric("Predicted Monthly Charges", f"{predicted_monthlycharges:.2f}")
            st.metric("Churn", churn_label)