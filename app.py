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

# Define constants for model file paths
MODEL_PATH_REGRESSION = "regression_model.pkl"
MODEL_PATH_CLASSIFICATION = "classification_model.pkl"

# Streamlit App Title
st.title("Copper Industry ML Model")

# File Upload
uploaded_file = st.file_uploader("Upload Copper Dataset (CSV)", type="csv")

if uploaded_file:
    # Load Dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Basic Cleaning
    if 'Material_Reference' in df.columns:
        df['Material_Reference'] = df['Material_Reference'].replace('00000', np.nan)

    # Identify Numeric and Categorical Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()

    # Select Target Variable
    if 'selling_price' in df.columns:
        target_variable = 'selling_price'
    elif 'status' in df.columns:
        target_variable = 'status'
    else:
        st.error("The dataset must contain either 'selling_price' or 'status' for regression or classification.")
        st.stop()

    # Choose Task
    task = st.radio("Choose a task", ("Regression", "Classification"))

    if task == "Regression" and target_variable == 'selling_price':
        # Prepare Data for Regression
        df = df.dropna(subset=[target_variable])
        df[target_variable] = np.log1p(df[target_variable])  # Log-transform target to handle skewness

        # Split Features and Target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

    elif task == "Classification" and target_variable == 'status':
        # Prepare Data for Classification
        df = df[df[target_variable].isin(['WON', 'LOST'])]
        df[target_variable] = df[target_variable].map({'WON': 1, 'LOST': 0})

        # Split Features and Target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

    else:
        st.error("Invalid task selected or target variable mismatch.")
        st.stop()

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Validate Columns
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]

    if not numeric_cols and not categorical_cols:
        st.error("No valid features found for preprocessing. Check your dataset.")
        st.stop()

    # Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Keeps untouched columns
    )

    # Model Pipeline
    if task == "Regression":
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        model_path = MODEL_PATH_REGRESSION
    else:
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        model_path = MODEL_PATH_CLASSIFICATION

    # Train Model
    try:
        model.fit(X_train, y_train)
        with open(model_path, 'wb') as f:
            pickle.dump((model, X_train.columns), f)
        st.success(f"{task} model trained and saved successfully!")
    except Exception as e:
        st.error(f"Error during training: {e}")
        st.stop()

    # Prediction Section
    st.header("Make Predictions")
    user_input = {}
    for col in X_train.columns:
        user_input[col] = st.text_input(f"Enter {col}", "")

    if st.button("Predict"):
        try:
            # Prepare Input Data
            input_df = pd.DataFrame([user_input])
            input_df = input_df.astype(X_train.dtypes)  # Match data types

            # Load Model
            with open(model_path, 'rb') as f:
                loaded_model, model_features = pickle.load(f)

            # Ensure Input Columns Match Model Features
            input_df = input_df.reindex(columns=model_features, fill_value=np.nan)

            # Predict
            prediction = loaded_model.predict(input_df)

            # Display Results
            if task == "Regression":
                st.success(f"Predicted Selling Price: ${np.expm1(prediction[0]):.2f}")
            else:
                status = "WON" if prediction[0] == 1 else "LOST"
                st.success(f"Predicted Status: {status}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Please upload a dataset to proceed.")
