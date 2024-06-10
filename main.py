import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
@st.cache
def load_data():
    # Load your historical data here
    # Example: data = pd.read_excel("your_data.xlsx")
    data = pd.read_excel("NIFTY50_JAN2021_APR2024.xlsx")
    return data

def preprocess_data(data):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Add extra columns for day and month
    data['Day'] = data['Date'].dt.day_name().str[:3]  # Get first three characters for day abbreviation
    data['Month'] = data['Date'].dt.month_name().str[:3]  # Get first three characters for month abbreviation

    # Add preprocessing steps here if needed
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
    st.title('Stock Market Prediction App')

    # Load data
    data = load_data()

    # Preprocess data
    data = preprocess_data(data)

    # Split data into features and targets
    X = data.drop(columns=['Close', 'Open Points'])
    y_close = data['Close']
    y_open_points = data['Open Points']

    # Train-test split
    X_train, X_test, y_train_close, y_test_close = train_test_split(X, y_close, test_size=0.2, random_state=42)
    _, _, y_train_open_points, y_test_open_points = train_test_split(X, y_open_points, test_size=0.2, random_state=42)

    # Model training for Close price prediction
    model_close = build_model(X_train, y_train_close)

    # Model evaluation for Close price prediction
    mse_close = evaluate_model(model_close, X_test, y_test_close)

    # Model training for Open Points prediction
    model_open_points = build_model(X_train, y_train_open_points)

    # Model evaluation for Open Points prediction
    mse_open_points = evaluate_model(model_open_points, X_test, y_test_open_points)

    st.write('Close Price Prediction - Mean Squared Error:', mse_close)
    st.write('Open Points Prediction - Mean Squared Error:', mse_open_points)

    # Form for user input
    st.sidebar.title('Select Prediction')
    prediction_type = st.sidebar.radio("Choose prediction type", ('Close Price', 'Open Points'))

    if prediction_type == 'Close Price':
        st.sidebar.write('You selected Close Price Prediction')
        features = {
            'Open': st.sidebar.number_input('Open', min_value=0.0),
            'ADVANCE / DECLINE RATIO': st.sidebar.number_input('Advance/Decline Ratio', min_value=0.0),
            'INDIAVIX Open': st.sidebar.number_input('INDIAVIX Open', min_value=0.0),
            'INDIAVIX Close': st.sidebar.number_input('INDIAVIX Close', min_value=0.0)
        }
        prediction = model_close.predict(pd.DataFrame([features]))
        st.write('Close Price Prediction:', prediction)

    elif prediction_type == 'Open Points':
        st.sidebar.write('You selected Open Points Prediction')
        features = {
            'Open': st.sidebar.number_input('Open', min_value=0.0),
            'ADVANCE / DECLINE RATIO': st.sidebar.number_input('Advance/Decline Ratio', min_value=0.0),
            'Close': st.sidebar.number_input('Close', min_value=0.0),
            'INDIAVIX Open': st.sidebar.number_input('INDIAVIX Open', min_value=0.0),
            'INDIAVIX Close': st.sidebar.number_input('INDIAVIX Close', min_value=0.0)
        }
        prediction = model_open_points.predict(pd.DataFrame([features]))
        st.write('Open Points Prediction:', prediction)

if __name__ == "__main__":
    main()
