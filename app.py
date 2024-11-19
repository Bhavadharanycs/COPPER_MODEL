import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pickle

# App Title
st.set_page_config(page_title="Copper ML App", layout="wide")
st.title("🔧 Copper Industry ML Application")

# Navigation
menu = st.sidebar.radio("Navigation", ["Upload Data", "Train Model", "Make Predictions"])

# Helper Functions
def clean_data(df, target):
    """Clean data and ensure target is numeric."""
    X = df.drop(columns=[target])
    y = df[target]

    # Handle missing values in features
    imputer = SimpleImputer(strategy='most_frequent')
    X_cleaned = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Convert all non-numeric columns to numeric
    for col in X_cleaned.select_dtypes(include=['object', 'category']).columns:
        X_cleaned[col] = LabelEncoder().fit_transform(X_cleaned[col])

    # Ensure target is numeric
    if y.dtype == object or y.dtype == str:
        y = LabelEncoder().fit_transform(y)
    else:
        y = pd.to_numeric(y, errors='coerce')

    # Handle missing values in the target variable
    y = pd.Series(SimpleImputer(strategy="most_frequent").fit_transform(y.values.reshape(-1, 1)).ravel())

    return X_cleaned, y

# Upload Data
if menu == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Save the dataset in session state for reuse
        st.session_state["dataset"] = df
        st.success("Dataset uploaded successfully!")

# Train Model
elif menu == "Train Model":
    st.header("Train Your Model")

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset first!")
    else:
        df = st.session_state["dataset"]
        target = st.selectbox("Select Target Variable", df.columns)

        if st.button("Train Model"):
            try:
                # Data Cleaning
                X, y = clean_data(df, target)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Decide task type
                if len(np.unique(y)) > 10:  # Regression if target has many unique values
                    model = RandomForestRegressor(random_state=42)
                    task = "Regression"
                else:  # Classification if target has few unique values
                    model = RandomForestClassifier(random_state=42)
                    task = "Classification"

                # Train the model
                model.fit(X_train, y_train)

                # Save the trained model
                model_file = f"{task.lower()}_model.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)

                st.success(f"{task} model trained successfully and saved as '{model_file}'!")
            except Exception as e:
                st.error(f"Training failed: {e}")

# Make Predictions
elif menu == "Make Predictions":
    st.header("Make Predictions Using Your Model")

    model_file = st.file_uploader("Upload Your Trained Model (.pkl)", type="pkl")
    if model_file:
        try:
            model = pickle.load(model_file)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    if "dataset" in st.session_state:
        df = st.session_state["dataset"]
        st.write("### Provide Input for Prediction")
        inputs = {}
        for col in df.columns:
            if col != st.session_state["target"]:  # Skip target column
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
