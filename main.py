import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel("NIFTY50_JAN2021_APR2024.xlsx")
    return data

def preprocess_data(data):
    # Select the relevant columns and drop any rows with missing values
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'INDIAVIX Open', 'INDIAVIX High', 'INDIAVIX Low', 'INDIAVIX Close']].dropna()
    # Extract Expiry Day from Date column
    data['Expiry Day'] = data['Date'].dt.dayofweek == 3  # 3 corresponds to Thursday
    data['Expiry Day'] = data['Expiry Day'].astype(int)  # Convert boolean to integer (1 or 0)
    
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
    feature_cols = ['Open', 'High', 'Low', 'INDIAVIX Open', 'INDIAVIX High', 'INDIAVIX Low', 'INDIAVIX Close', 'Expiry Day']
    target_col = 'Close'

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Expiry Day'])
        ],
        remainder='passthrough'
    )

    # Append classifier to preprocessing pipeline
    # Now we have a full prediction pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Model training
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

    expiry_day = st.radio("Expiry Day", ['No', 'Yes'])
    if expiry_day == 'Yes':
        expiry_day = 1
    else:
        expiry_day = 0

    input_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high],
        'Low': [low],
        'INDIAVIX Open': [india_vix_open],
        'INDIAVIX High': [india_vix_high],
        'INDIAVIX Low': [india_vix_low],
        'INDIAVIX Close': [india_vix_close],
        'Expiry Day': [expiry_day]
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write('Close Price Prediction:', prediction[0])

if __name__ == "__main__":
    main()
