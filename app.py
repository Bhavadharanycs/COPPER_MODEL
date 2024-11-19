import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# App Title
st.set_page_config(page_title="Copper Industry ML Model", layout="centered")
st.title("Copper Industry ML Model")
st.sidebar.title("Navigation")
st.sidebar.markdown("Select an option from the menu below.")

# Navigation Sidebar
menu = st.sidebar.radio("Choose an option:", ["Home", "Train Model", "Make Predictions"])

# Global Variables
MODEL_PATH = "saved_model.pkl"

if menu == "Home":
    st.markdown("""
    ## Welcome to the Copper Industry ML Model App
    Use this application to:
    1. Train a machine learning model for regression or classification tasks using your copper dataset.
    2. Make predictions with a trained model.
    
    **Instructions:**
    - Upload a dataset containing either `selling_price` (for regression) or `status` (for classification).
    - Train the model in the "Train Model" section.
    - Use the "Make Predictions" section to test the model with new inputs.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d8/Copper_tube.jpg", use_column_width=True)

elif menu == "Train Model":
    st.header("Train Your Model")

    uploaded_file = st.file_uploader("Upload Copper Dataset (CSV)", type="csv")

    if uploaded_file:
        # Load Dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Cleaning and Column Identification
        if 'Material_Reference' in df.columns:
            df['Material_Reference'] = df['Material_Reference'].replace('00000', np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=[object]).columns.tolist()

        # Check for Target Variable
        target_variable = None
        if 'selling_price' in df.columns:
            target_variable = 'selling_price'
        elif 'status' in df.columns:
            target_variable = 'status'
        else:
            st.error("Dataset must contain either 'selling_price' or 'status'.")
            st.stop()

        task = st.radio("Choose a task:", ["Regression", "Classification"], horizontal=True)

        if task == "Regression" and target_variable == 'selling_price':
            # Data Preparation
            df = df.dropna(subset=[target_variable])
            df[target_variable] = np.log1p(df[target_variable])
            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Preprocessing and Model Pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(random_state=42))
            ])
            model.fit(X_train, y_train)

            # Save Model
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump((model, X.columns), f)

            st.success("Regression model trained and saved successfully!")

        elif task == "Classification" and target_variable == 'status':
            # Data Preparation
            df = df[df[target_variable].isin(['WON', 'LOST'])]
            df[target_variable] = df[target_variable].map({'WON': 1, 'LOST': 0})
            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Preprocessing and Model Pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            model.fit(X_train, y_train)

            # Save Model
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump((model, X.columns), f)

            st.success("Classification model trained and saved successfully!")

elif menu == "Make Predictions":
    st.header("Make Predictions")

    if os.path.exists(MODEL_PATH):
        # Load Saved Model
        with open(MODEL_PATH, 'rb') as f:
            loaded_model, feature_columns = pickle.load(f)

        user_input = {}
        for col in feature_columns:
            user_input[col] = st.text_input(f"Enter {col}:", "")

        if st.button("Predict"):
            try:
                # Prepare Input for Prediction
                input_df = pd.DataFrame([user_input])
                input_df = input_df.astype(dict(zip(feature_columns, loaded_model.named_steps['preprocessor'].transformers_[0][1].named_steps['imputer'].statistics_)))

                # Make Prediction
                prediction = loaded_model.predict(input_df)

                # Show Results
                if isinstance(loaded_model.named_steps['preprocessor'], RandomForestRegressor):
                    st.success(f"Predicted Selling Price: ${np.expm1(prediction[0]):.2f}")
                else:
                    status = "WON" if prediction[0] == 1 else "LOST"
                    st.success(f"Predicted Status: {status}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("No model found. Please train a model first in the 'Train Model' section.")
