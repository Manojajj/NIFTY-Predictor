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
    # Drop any non-numeric columns or handle them appropriately
    data = data.select_dtypes(include=[float, int])
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
    st.title("NIFTY 50 Predictor")

    data = load_data()
    
    # Preprocess data
    data = preprocess_data(data)
    
    if data.empty:
        st.error("The dataset is empty after preprocessing.")
        return

    # Define feature columns and target columns
    feature_cols = data.columns.drop(['Close', 'Open'])  # Adjust according to your dataset
    target_col_close = 'Close'
    target_col_open = 'Open'

    X = data[feature_cols]
    y_close = data[target_col_close]
    y_open = data[target_col_open]

    X_train, X_test, y_train_close, y_test_close = train_test_split(X, y_close, test_size=0.2, random_state=42)
    _, _, y_train_open_points, y_test_open_points = train_test_split(X, y_open, test_size=0.2, random_state=42)

    # Model training for Close price prediction
    model_close = build_model(X_train, y_train_close)
    # Model evaluation for Close price prediction
    mse_close = evaluate_model(model_close, X_test, y_test_close)
    st.write(f"Mean Squared Error for Close Price Prediction: {mse_close}")

    # Model training for Open points prediction
    model_open = build_model(X_train, y_train_open_points)
    # Model evaluation for Open points prediction
    mse_open = evaluate_model(model_open, X_test, y_test_open_points)
    st.write(f"Mean Squared Error for Open Points Prediction: {mse_open}")

    # Make predictions for new data
    st.write("Make Predictions")
    new_data = st.text_area("Enter new data in CSV format", value="")  # Enter new data as CSV format
    if st.button("Predict"):
        if new_data:
            new_data_df = pd.read_csv(pd.compat.StringIO(new_data))
            new_data_df = preprocess_data(new_data_df)
            if not new_data_df.empty:
                prediction_close = model_close.predict(new_data_df)
                prediction_open = model_open.predict(new_data_df)
                st.write('Close Price Prediction:', prediction_close)
                st.write('Open Points Prediction:', prediction_open)
            else:
                st.error("The input data is invalid after preprocessing.")
        else:
            st.error("No data entered.")

if __name__ == "__main__":
    main()
