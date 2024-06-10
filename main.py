import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel("NIFTY50_JAN2021_APR2024.xlsx")
    return data

def preprocess_data(data):
    # Select the relevant columns and drop any rows with missing values
    data = data[['Open', 'High', 'Low', 'Close', 'INDIAVIX Open', 'INDIAVIX High', 'INDIAVIX Low', 'INDIAVIX Close', 'Date']]
    data['Expiry'] = data['Date'].dt.dayofweek  # 0: Monday, 1: Tuesday, ..., 6: Sunday
    data['Expiry'] = data['Expiry'].apply(lambda x: 1 if x == 3 else 0) # Thursday is considered as expiry day
    data.drop(columns=['Date'], inplace=True)
    return data

def build_model():
    model = LinearRegression()
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
    feature_cols = ['Open', 'High', 'Low', 'INDIAVIX Open', 'INDIAVIX High', 'INDIAVIX Low', 'INDIAVIX Close', 'Expiry']
    target_col = 'Close'

    X = data[feature_cols]
    y = data[target_col]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model training
    model = build_model()
    model.fit(X_train, y_train)

    # Model evaluation
    mse = evaluate_model(model, X_test, y_test)
    st.write(f"Mean Squared Error for Close Price Prediction: {mse}")

    # Make predictions for new data
    st.write("Make Predictions")

    open_price = st.number_input("Open")
    high = st.number_input("High")
    low = st.number_input("Low")
    india_vix_open = st.number_input("INDIAVIX Open")
    india_vix_high = st.number_input("INDIAVIX High")
    india_vix_low = st.number_input("INDIAVIX Low")
    india_vix_close = st.number_input("INDIAVIX Close")
    expiry_day = st.selectbox("Expiry Day", ['Yes', 'No'])

    expiry = 1 if expiry_day == 'Yes' else 0

    input_data = scaler.transform([[open_price, high, low, india_vix_open, india_vix_high, india_vix_low, india_vix_close, expiry]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write('Close Price Prediction:', prediction[0])

if __name__ == "__main__":
    main()
