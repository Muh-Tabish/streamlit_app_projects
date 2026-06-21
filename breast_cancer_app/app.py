import pickle
import numpy as np
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_saved_files():
    base_dir = Path(__file__).parent

    model_path = base_dir / "models" / "breast_cancer_model.pkl"
    scaler_path = base_dir / "models" / "scaler.pkl"
    features_path = base_dir / "models" / "features.pkl"

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    with open(features_path, "rb") as file:
        features = pickle.load(file)

    return model, scaler, features


# -----------------------------
# Make Prediction
# -----------------------------
def predict_cancer(model, scaler, user_values):
    input_array = np.array(user_values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    return prediction, probability


# -----------------------------
# App UI
# -----------------------------
model, scaler, features = load_saved_files()

st.title("🩺 Breast Cancer Prediction App")
st.write(
    "This app predicts whether a breast tumor is **Benign** or **Malignant** using a Machine Learning model."
)

st.info("Model Used: Logistic Regression | Dataset: Breast Cancer Wisconsin Dataset")

# Sidebar
st.sidebar.header("Patient Tumor Measurements")

feature_labels = {
    "mean radius": "Tumor Radius",
    "mean texture": "Tumor Texture",
    "mean perimeter": "Tumor Perimeter",
    "mean area": "Tumor Area",
    "mean smoothness": "Tumor Smoothness"
}

sample_patient = st.sidebar.selectbox(
    "Choose Sample Patient",
    ["Custom Input", "Sample Malignant", "Sample Benign"]
)

if sample_patient == "Sample Malignant":
    default_values = [17.99, 10.38, 122.8, 1001.0, 0.1184]

elif sample_patient == "Sample Benign":
    default_values = [11.42, 20.38, 77.58, 386.1, 0.1425]

else:
    default_values = [0.0, 0.0, 0.0, 0.0, 0.0]


user_values = []

for index, feature in enumerate(features):
    value = st.sidebar.number_input(
        feature_labels[feature],
        value=float(default_values[index]),
        step=0.01
    )
    user_values.append(value)


# Main Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Features Used", len(features))

with col2:
    st.metric("Model Type", "Logistic Regression")

with col3:
    st.metric("Output Classes", "2")


st.subheader("Entered Patient Values")

patient_data = {
    feature_labels[feature]: user_values[index]
    for index, feature in enumerate(features)
}

st.table(patient_data)


if st.button("Predict Cancer Type"):
    prediction, probability = predict_cancer(
        model,
        scaler,
        user_values
    )

    malignant_prob = probability[0]
    benign_prob = probability[1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Benign Tumor")
    else:
        st.error("Prediction: Malignant Tumor")

    st.write("Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Malignant Probability",
            f"{malignant_prob * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Benign Probability",
            f"{benign_prob * 100:.2f}%"
        )

    st.progress(float(max(probability)))


with st.expander("About This Project"):
    st.write("""
    This project uses Logistic Regression to classify breast tumors as benign or malignant.
    The model was trained on selected important features from the Breast Cancer Wisconsin Dataset.
    
    This app demonstrates:
    
    - Machine Learning model training
    - Feature scaling
    - Model saving using Pickle
    - Streamlit UI development
    - Real-time prediction
    - Basic deployment-ready project structure
    """)

st.warning(
    "Disclaimer: This app is for educational purposes only and should not be used for real medical diagnosis."
)