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

# Title
st.title("Copper Industry ML Model")

# File Upload
uploaded_file = st.file_uploader("Upload Copper Dataset (CSV)", type="csv")

if uploaded_file:
    # Load Dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Initial Cleaning
    if 'Material_Reference' in df.columns:
        df['Material_Reference'] = df['Material_Reference'].replace('00000', np.nan)

    # Identify Column Types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()

    if 'selling_price' in df.columns:
        target_variable = 'selling_price'
    elif 'Status' in df.columns:
        target_variable = 'Status'
    else:
        st.error("The dataset must contain either 'Selling_Price' or 'Status' for regression or classification.")
        st.stop()

    # Select Task
    task = st.radio("Choose a task", ("Regression", "Classification"))

    # Preprocess Data
    if task == "Regression" and target_variable == 'Selling_Price':
        # Drop rows with missing target
        df = df.dropna(subset=[target_variable])

        # Log-transform target to handle skewness
        df[target_variable] = np.log1p(df[target_variable])

        # Split Features and Target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing Pipeline
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

        # Model Pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Fit Model
        model.fit(X_train, y_train)

        # Save Model
        with open('regression_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        st.success("Regression model trained and saved!")

    elif task == "Classification" and target_variable == 'Status':
        # Keep only 'WON' and 'LOST'
        df = df[df[target_variable].isin(['WON', 'LOST'])]

        # Map target to binary
        df[target_variable] = df[target_variable].map({'WON': 1, 'LOST': 0})

        # Split Features and Target
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing Pipeline
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

        # Model Pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Fit Model
        model.fit(X_train, y_train)

        # Save Model
        with open('classification_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        st.success("Classification model trained and saved!")

    # Ensure X is defined before prediction
if 'X' in locals() and not X.empty:
    st.header("Make Predictions")
    user_input = {}

    # Iterate through feature columns
    for col in X.columns:
        user_input[col] = st.text_input(f"Enter {col}", "")

    if st.button("Predict"):
        # Prepare Input for Prediction
        try:
            input_df = pd.DataFrame([user_input])
            input_df = input_df.astype(X.dtypes)

            # Load Model
            model_file = 'regression_model.pkl' if task == "Regression" else 'classification_model.pkl'
            with open(model_file, 'rb') as f:
                loaded_model = pickle.load(f)

            # Predict
            prediction = loaded_model.predict(input_df)

            # Show Results
            if task == "Regression":
                st.success(f"Predicted Selling Price: ${np.expm1(prediction[0]):.2f}")
            else:
                status = "WON" if prediction[0] == 1 else "LOST"
                st.success(f"Predicted Status: {status}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Dataset must be loaded and processed before making predictions.")

 
