import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# App Title
st.set_page_config(page_title="Copper Industry ML App", layout="wide")
st.title("📊 Copper Industry ML Model")
st.markdown("""
    Welcome to the **Copper Industry ML Application**! 
    Upload a dataset, train a machine learning model, and make predictions. 
    Use this app for either regression or classification tasks.
""")

# Sidebar
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Upload Data", "Train Model", "Make Predictions", "About"])

# Global Variables
uploaded_file = None
X_train, X_test, y_train, y_test, model, task, numeric_cols, categorical_cols = (None,) * 8


# Helper Function for Preprocessing
def preprocess_data(df, target):
    global numeric_cols, categorical_cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    return numeric_cols, categorical_cols


# Upload Data
if menu == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload Copper Dataset (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.write("### Dataset Preview")
        st.write(df.head())
        st.write("### Dataset Info")
        st.write(df.describe())

# Train Model
elif menu == "Train Model":
    st.header("Train Model")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_variable = st.selectbox("Select Target Variable", df.columns)

        if target_variable:
            numeric_cols, categorical_cols = preprocess_data(df, target_variable)

            task = st.radio("Select Task", ["Regression", "Classification"])
            st.write(f"Selected Task: **{task}**")

            if task == "Regression" and df[target_variable].dtype in ['int64', 'float64']:
                df = df.dropna(subset=[target_variable])
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
                    ]
                )
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(random_state=42))
                ])
                model.fit(X_train, y_train)

                with open('regression_model.pkl', 'wb') as f:
                    pickle.dump(model, f)

                st.success("Regression model trained successfully!")

            elif task == "Classification" and len(df[target_variable].unique()) == 2:
                df = df.dropna(subset=[target_variable])
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
                    ]
                )
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42))
                ])
                model.fit(X_train, y_train)

                with open('classification_model.pkl', 'wb') as f:
                    pickle.dump(model, f)

                st.success("Classification model trained successfully!")

            else:
                st.error("Invalid task or target variable type. Please check your dataset.")

    else:
        st.warning("Please upload a dataset first.")

# Make Predictions
elif menu == "Make Predictions":
    st.header("Make Predictions")
    if uploaded_file and model:
        st.write("### Enter Feature Values for Prediction")

        user_input = {}
        for col in numeric_cols + categorical_cols:
            user_input[col] = st.text_input(f"{col}", "")

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([user_input])
                input_df = input_df.astype({col: df[col].dtype for col in df.columns if col in input_df})
                model_file = 'regression_model.pkl' if task == "Regression" else 'classification_model.pkl'
                with open(model_file, 'rb') as f:
                    loaded_model = pickle.load(f)
                prediction = loaded_model.predict(input_df)

                if task == "Regression":
                    st.success(f"Predicted Value: {prediction[0]:.2f}")
                else:
                    result = "Positive Class" if prediction[0] == 1 else "Negative Class"
                    st.success(f"Predicted Class: {result}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please upload data and train a model first.")

# About
elif menu == "About":
    st.header("About this App")
    st.markdown("""
    - **Developer:** Your Name
    - **Purpose:** Machine Learning application for the copper industry.
    - **Features:** 
      - Upload dataset
      - Train regression or classification models
      - Make predictions
    """)

