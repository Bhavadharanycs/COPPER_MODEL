import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle

# App Title
st.set_page_config(page_title="Copper ML App", layout="wide")
st.title("🔧 Copper Industry ML Application")

# Navigation
menu = st.sidebar.selectbox("Menu", ["Upload Data", "Train Model", "Predict"])

# Helper Function
def clean_data(df, target):
    """Clean data by handling missing values and identifying columns."""
    X = df.drop(columns=[target])
    y = df[target]

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_cleaned = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_cleaned, y

# Upload Data
if menu == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Store dataset in session state for reuse
        st.session_state["dataset"] = df
        st.success("Dataset uploaded and saved!")

# Train Model
elif menu == "Train Model":
    st.header("Train a Model")

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset first!")
    else:
        df = st.session_state["dataset"]
        target = st.selectbox("Select Target Variable", df.columns)

        if st.button("Train Model"):
            try:
                # Clean data
                X, y = clean_data(df, target)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Determine task based on target type
                if y.dtype == np.number:
                    model = RandomForestRegressor(random_state=42)
                    task = "regression"
                else:
                    model = RandomForestClassifier(random_state=42)
                    task = "classification"

                model.fit(X_train, y_train)

                # Save model
                model_file = f"{task}_model.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)

                st.success(f"{task.capitalize()} model trained and saved as '{model_file}'!")
            except Exception as e:
                st.error(f"Training failed: {e}")

# Predict
elif menu == "Predict":
    st.header("Make Predictions")

    model_file = st.text_input("Enter Model Filename (e.g., regression_model.pkl)", "")
    if model_file and st.button("Load Model"):
        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            st.success(f"Model '{model_file}' loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    if model_file and "dataset" in st.session_state:
        df = st.session_state["dataset"]
        inputs = {}
        for col in df.columns:
            if col != model.target_names[0]:
                inputs[col] = st.text_input(f"Enter value for {col}")

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([inputs])
                prediction = model.predict(input_df)
                st.success(f"Prediction: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please upload a dataset and train a model first!")
