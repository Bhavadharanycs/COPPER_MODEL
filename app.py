import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import streamlit as st

# Load dataset
df = pd.read_csv("Copper_Set 1.csv")

# Data Understanding & Cleaning
categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()

# Handle rubbish values
if 'Material_Reference' in df.columns:
    df['Material_Reference'] = df['Material_Reference'].replace('00000', np.nan)

# Drop unnecessary columns
if 'INDEX' in df.columns:
    df.drop(columns=['INDEX'], inplace=True)

# Fill missing values quickly
df[continuous_vars] = df[continuous_vars].fillna(df[continuous_vars].mean())
df[categorical_vars] = df[categorical_vars].fillna(df[categorical_vars].mode().iloc[0])

# Encode categorical variables (label encoding for simplicity)
encoder = LabelEncoder()
for col in categorical_vars:
    df[col] = encoder.fit_transform(df[col])

# Split data into features and target
if 'Selling_Price' in df.columns:  # Regression
    target_col = 'Selling_Price'
    model = RandomForestRegressor(random_state=42, n_estimators=50)  # Reduce estimators for faster training
else:  # Classification
    target_col = 'Status'
    model = RandomForestClassifier(random_state=42, n_estimators=50)

X = df.drop(columns=[target_col])
y = df[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save model and preprocessing steps
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Streamlit GUI
st.title("Optimized ML Model GUI")

# Task selection
task = st.selectbox("Select Task", ["Regression", "Classification"])

# Input fields for new data
st.subheader("Enter Input Features:")
input_data = []
for col in X.columns:
    value = st.text_input(f"Enter value for {col}:")
    input_data.append(float(value) if value else 0.0)  # Default to 0.0 if no input

# Predict button
if st.button("Predict"):
    # Convert input to scaled format
    input_df = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_df)

    # Predict
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(input_scaled)
    st.write(f"Prediction: {prediction[0]}")
