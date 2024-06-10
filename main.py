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
    new_data = st.text_area("Enter new data in CSV format", value="")  # Enter new data as CSV format
    if st.button("Predict"):
        if new_data:
            try:
                new_data_df = pd.read_csv(pd.compat.StringIO(new_data))
                new_data_df = preprocess_data(new_data_df)
                if not new_data_df.empty:
                    prediction = model.predict(new_data_df[feature_cols])
                    st.write('Close Price Prediction:', prediction)
                else:
                    st.error("The input data is invalid after preprocessing.")
            except Exception as e:
                st.error(f"Error processing input data: {e}")
        else:
            st.error("No data entered.")

if __name__ == "__main__":
    main()
