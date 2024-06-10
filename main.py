import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    data = data[['Date', 'Open Points', 'Open', 'High', 'Low', 'Close', 'ADVANCES', 'DECLINES', 
                 'INDIAVIX Open', 'INDIAVIX High', 'INDIAVIX Low', 'INDIAVIX Close']].dropna()
    # Extract Day and Month from Date column
    data['Day'] = data['Date'].dt.dayofweek  # 0: Monday, 1: Tuesday, ..., 6: Sunday
    data['Month'] = data['Date'].dt.month
    return data

def build_model():
    model = RandomForestRegressor()
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
    feature_cols = ['Open Points', 'Open', 'High', 'Low', 'INDIAVIX Open', 
                    'INDIAVIX High', 'INDIAVIX Low', 'INDIAVIX Close', 'Day', 'Month']
    target_col = 'Close'

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps for categorical features
    categorical_features = ['Day', 'Month']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps for all features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ])

    # Append classifier to preprocessing pipeline
    # Now we have a full prediction pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # Model training
    model.fit(X_train, y_train)
    # Model evaluation
    mse = evaluate_model(model, X_test, y_test)
    st.write(f"Mean Squared Error for Close Price Prediction: {mse}")

    # Make predictions for new data
    st.write("Make Predictions")

    open_points = st.number_input("Open Points")
    open_price = st.number_input("Open")
    high = st.number_input("High")
    low = st.number_input("Low")
    india_vix_open = st.number_input("INDIAVIX Open")
    india_vix_high = st.number_input("INDIAVIX High")
    india_vix_low = st.number_input("INDIAVIX Low")
    india_vix_close = st.number_input("INDIAVIX Close")
    day = st.selectbox("Day", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    month = st.selectbox("Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    input_data = pd.DataFrame({
        'Open Points': [open_points],
        'Open': [open_price],
        'High': [high],
        'Low': [low],
        'INDIAVIX Open': [india_vix_open],
        'INDIAVIX High': [india_vix_high],
        'INDIAVIX Low': [india_vix_low],
        'INDIAVIX Close': [india_vix_close],
        'Day': [['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].index(day)],
        'Month': [['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(month)]
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write('Close Price Prediction:', prediction[0])

if __name__ == "__main__":
    main()
