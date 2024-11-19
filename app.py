import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Title and Description
st.title("Copper Industry ML Predictions")
st.write("Select a task (Regression or Classification) and input values to predict the outcome.")

# Task Selection
task = st.radio("Select Task", ("Regression", "Classification"))

# Load Preprocessed Data for Schema Reference
@st.cache_data
def load_data():
    # Simulated data structure (update with real preprocessed schema if available)
    data = pd.read_csv("Copper_Set.csv")  # Replace with your preprocessed dataset
    return data

data = load_data()

# Separate Features and Target
if task == "Regression":
    target_variable = "Selling_Price"
    if target_variable not in data.columns:
        st.error(f"'{target_variable}' column not found in dataset.")
else:
    target_variable = "Status"
    if target_variable not in data.columns:
        st.error(f"'{target_variable}' column not found in dataset.")

features = [col for col in data.columns if col != target_variable]

# User Input for Feature Values
st.header("Input Feature Values")
user_input = {}

for feature in features:
    dtype = data[feature].dtype
    if np.issubdtype(dtype, np.number):
        user_input[feature] = st.number_input(f"Enter {feature}", value=float(data[feature].mean()))
    else:
        user_input[feature] = st.text_input(f"Enter {feature}", value=str(data[feature].mode()[0]))

# Convert Input to DataFrame
input_df = pd.DataFrame([user_input])

# Feature Engineering and Transformations
@st.cache_data
def apply_transformations(input_data, task):
    # Apply transformations used in model training
    input_data = input_data.copy()
    # Scaling for numerical columns
    scaler = pickle.load(open("scaler.pkl", "rb"))
    numeric_cols = input_data.select_dtypes(include=[np.number]).columns
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Encoding for categorical columns
    encoder = pickle.load(open("encoder.pkl", "rb"))
    categorical_cols = input_data.select_dtypes(include=[object]).columns
    input_data[categorical_cols] = encoder.transform(input_data[categorical_cols])

    return input_data

try:
    transformed_input = apply_transformations(input_df, task)
except Exception as e:
    st.error(f"Transformation Error: {e}")

# Load Model and Make Prediction
if st.button("Predict"):
    try:
        model_file = "regression_model.pkl" if task == "Regression" else "classification_model.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        prediction = model.predict(transformed_input)

        if task == "Regression":
            # Reverse log transformation if applied
            st.success(f"Predicted Selling Price: ${np.expm1(prediction[0]):.2f}")
        else:
            # Display classification outcome
            status = "WON" if prediction[0] == 1 else "LOST"
            st.success(f"Predicted Status: {status}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
