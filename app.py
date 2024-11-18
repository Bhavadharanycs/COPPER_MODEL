import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox
import pickle

# Load dataset
@st.cache
def load_data():
    data_path = '/mnt/data/Copper_Set 1.csv'  # Update based on deployment
    return pd.read_csv(data_path)

df = load_data()

# Sidebar for task selection
task = st.sidebar.selectbox("Select Task", ["Regression (Selling Price)", "Classification (Lead Status)"])

# Data Cleaning and Preprocessing
st.title("Copper Industry ML Tool")

# Handling invalid values
df['Material_Reference'] = df['Material_Reference'].replace('00000', np.nan)
df.drop(columns=['INDEX'], inplace=True, errors='ignore')

# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
df.iloc[:, :] = imputer.fit_transform(df)

# Treating skewness
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col])

# Treating outliers
def handle_outliers(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower_bound, upper_bound)

handle_outliers(df, numeric_cols)

# Encoding categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Feature-target splitting
if task == "Regression (Selling Price)":
    target = "Selling_Price"
elif task == "Classification (Lead Status)":
    target = "Status"
    df = df[df[target].isin(['WON', 'LOST'])]  # Keep only relevant status

X = df.drop(columns=[target])
y = df[target]

# Model Training
st.write("Training Models...")

if task == "Regression (Selling Price)":
    y = np.log1p(y)  # Transform target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
elif task == "Classification (Lead Status)":
    y = encoder.fit_transform(y)  # Encode target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)
pickle.dump(model, open(f"{task.replace(' ', '_')}_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Model Evaluation
if task == "Regression (Selling Price)":
    y_pred = model.predict(X_test)
    y_pred = np.expm1(y_pred)  # Reverse transformation
    st.write(f"Regression RMSE: {np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))}")
elif task == "Classification (Lead Status)":
    y_pred = model.predict(X_test)
    st.write(f"Classification Accuracy: {accuracy_score(y_test, y_pred)}")
    st.text(classification_report(y_test, y_pred))

# Streamlit UI for Prediction
st.header("Make a Prediction")
user_input = {}
for col in X.columns:
    user_input[col] = st.text_input(col, "")

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_df = scaler.transform(input_df)  # Scale inputs
    prediction = model.predict(input_df)

    if task == "Regression (Selling Price)":
        prediction = np.expm1(prediction)  # Reverse log transform
    else:
        prediction = encoder.inverse_transform([int(prediction)])

    st.write(f"Prediction: {prediction[0]}")
