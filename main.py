import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel("NIFTY50_JAN2021_APR2024.xlsx")
    return data

def preprocess_data(data):
    # Select the relevant columns and drop any rows with missing values
    data = data[['Open Points', 'Open', 'ADVANCE / DECLINE RATIO', 'INDIA VIX Close', 'Close']].dropna()
    return data

def build_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def main():
    st.title("NIFTY 50 Close Price Predictor")

    data = load_data()
    
    # Preprocess data
    data = preprocess_data(data)
    
    if data.empty:
        st.error("The dataset is empty after preprocessing.")
        return

    # Define feature columns and target columns
    feature_cols = ['Open Points', 'Open', 'ADVANCE / DECLINE RATIO', 'INDIA VIX Close']
    target_col = 'Close'

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = build_model(X_train, y_train)
    # Model evaluation
    mse = evaluate_model(model, X_test, y_test)
    st.write(f"Mean Squared Error for Close Price Prediction: {mse}")

    # Make predictions for new data
    st.write("Make Predictions")

    open_points = st.number_input("Open Points (Open Price - Prev. Close)", value=float(data['Open Points'].mean()))
    open_price = st.number_input("Open", value=float(data['Open'].mean()))
    adv_dec_ratio = st.number_input("Advance / Decline Ratio", value=float(data['ADVANCE / DECLINE RATIO'].mean()))
    india_vix_close = st.number_input("INDIA VIX", value=float(data['INDIAVIX Close'].mean()))

    input_data = pd.DataFrame({
        'Open Points': [open_points],
        'Open': [open_price],
        'ADVANCE / DECLINE RATIO': [adv_dec_ratio],
        'INDIA VIX Close': [india_vix_close]
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write('Close Price Prediction:', prediction[0])

if __name__ == "__main__":
    main()
