import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import json

# Load dataset
file_path = r"C:\Users\budat\Desktop\projects\PAD project\Pad Project\survey lung cancer.csv"
df = pd.read_csv(file_path, delimiter="\t")

# Normalize column names (remove spaces & uppercase)
df.columns = df.columns.str.strip().str.upper()

# Check for required column
if 'LUNG_CANCER' not in df.columns:
    st.error("Error: 'LUNG_CANCER' column not found in dataset.")
    st.stop()

# Encode categorical variables
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' and col != 'GENDER':
        df[col] = encoder.fit_transform(df[col])

# Encode GENDER separately
df['GENDER'] = df['GENDER'].map({'Male': 1, 'Female': 0})

# Features and target
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# File to store logs
log_file = "user_inputs_log.json"

# Function to log inputs
def log_user_input(user_data, prediction):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'input': user_data,
        'prediction': int(prediction)
    }
    try:
        with open(log_file, "r") as file:
            logs = json.load(file)
    except FileNotFoundError:
        logs = []

    logs.append(log_entry)
    with open(log_file, "w") as file:
        json.dump(logs, file)

# Streamlit UI
st.title("🚭 Lung Cancer Prediction & Real-Time Analysis")
st.write("Enter your details to predict lung cancer risk based on symptoms and lifestyle.")

# Model accuracy
st.subheader("🔍 Model Evaluation")
st.write(f"**Accuracy:** `{accuracy:.2f}`")
st.text("Classification Report:")
st.text(report)

# Feature importance
st.subheader("📊 Feature Importance")
feature_importances = model.feature_importances_
feature_names = X.columns
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=feature_importances, y=feature_names, ax=ax)
ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# Input from user
def user_input():
    user_data = {}
    for feature in X.columns:
        if feature == "AGE":
            user_data[feature] = st.number_input("Age", min_value=10, max_value=100, value=50)
        elif feature == "GENDER":
            gender_input = st.selectbox("Gender", ['Male', 'Female'])
            user_data[feature] = 1 if gender_input == "Male" else 0
        else:
            choice = st.selectbox(feature.replace("_", " ").title(), ['Yes', 'No'], key=feature)
            user_data[feature] = 1 if choice == "Yes" else 0
    return np.array([list(user_data.values())]), user_data

# Get prediction
user_symptoms, user_data_dict = user_input()

if len(user_symptoms[0]) != X.shape[1]:
    st.error("Feature mismatch: Please recheck the inputs.")
else:
    user_symptoms_scaled = scaler.transform(user_symptoms)
    prediction = model.predict(user_symptoms_scaled)

    result = "⚠️ High Risk of Lung Cancer" if prediction[0] == 1 else "✅ Low Risk of Lung Cancer"
    st.subheader(f"🩺 Prediction Result: {result}")

    # Log input
    log_user_input(user_data_dict, prediction[0])

# Show recent logs
st.subheader("🕵️ Real-Time Pattern Analysis")
try:
    with open(log_file, "r") as file:
        logs = json.load(file)
        st.write(f"Number of predictions made: {len(logs)}")
        st.json(logs[-5:])  # Last 5 entries
except FileNotFoundError:
    st.write("No logs available yet.")
