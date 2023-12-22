import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression

lstm_model = load_model('lstm.h5')
log_reg_model = joblib.load('linear.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_input_lstm(input_data, scaler):
    scaled_input = scaler.transform(input_data)
    reshaped_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
    return reshaped_input

def predict_price_category_lstm(user_input_lstm, scaler):
    processed_input_lstm = preprocess_input_lstm(pd.DataFrame(user_input_lstm, index=[0]), scaler)
    predicted_category = np.argmax(lstm_model.predict(processed_input_lstm), axis=-1)
    return predicted_category

def predict_exact_price_logistic_regression(category, user_input_log_reg):
    processed_input_log_reg = pd.DataFrame(user_input_log_reg, index=[0])

    predicted_price = log_reg_model.predict(processed_input_log_reg)
    return predicted_price


if __name__ == "__main__":
    scaler = joblib.load('scaler.pkl')

    user_input_lstm = {
        'ODO': float(input("Enter ODO reading: ")),
        'Year': int(input("Enter Year: ")),
        'Age': float(input("Enter Age: ")),
        'Mileage': float(input("Enter Milage: ")),
        'Engine': float(input("Enter Engine: ")),
        'Power': float(input("Enter Power: ")),
        'Seats': float(input('Enter Seats: '))
    }

    fuel_choice = int(input("Choose Fuel type (0: Diesel, 1: Petrol): "))
    user_input_lstm['Fuel_Diesel'] = 1.0 if fuel_choice == 0 else 0.0
    user_input_lstm['Fuel_Petrol'] = 1.0 if fuel_choice == 1 else 0.0

    trans_choice = int(input("Choose Transmission type (0: Automatic, 1: Manual): "))
    user_input_lstm['Transmission_Automatic'] = 1.0 if trans_choice == 0 else 0.0
    user_input_lstm['Transmission_Manual'] = 1.0 if trans_choice == 1 else 0.0
    owner_choice = int(input("Choose Owner type (0: First, 1: Second, 2: Third, 3: Fourth & Above): "))
    if owner_choice == 0:
        user_input_lstm['Owner_First'] = 1.0
        user_input_lstm['Owner_Fourth & Above'] = 0.0
        user_input_lstm['Owner_Second'] = 0.0
        user_input_lstm['Owner_Third'] = 0.0
    elif owner_choice == 1:
        user_input_lstm['Owner_First'] = 0.0
        user_input_lstm['Owner_Fourth & Above'] = 0.0
        user_input_lstm['Owner_Second'] = 1.0
        user_input_lstm['Owner_Third'] = 0.0
    elif owner_choice == 2:
        user_input_lstm['Owner_First'] = 0.0
        user_input_lstm['Owner_Fourth & Above'] = 0.0
        user_input_lstm['Owner_Second'] = 0.0
        user_input_lstm['Owner_Third'] = 1.0
    elif owner_choice == 3:
        user_input_lstm['Owner_First'] = 0.0
        user_input_lstm['Owner_Fourth & Above'] = 1.0
        user_input_lstm['Owner_Second'] = 0.0
        user_input_lstm['Owner_Third'] = 0.0

    print("LSTM :")
    predicted_category = predict_price_category_lstm(user_input_lstm, scaler)
    print("Predicted Category:", predicted_category)
    print("Linear Regression :")
    predicted_price = predict_exact_price_logistic_regression(predicted_category, user_input_lstm)
    print("Predicted Price:", predicted_price)

# #To save models
# import joblib  # Import joblib to save the model
# #Save the model to a file
# joblib.dump(model, 'logistic.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# DL Model
# mode.save(lstm.h5)
