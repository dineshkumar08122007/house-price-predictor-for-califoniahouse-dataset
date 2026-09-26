import pandas as pd
import joblib
from preprocessing import StandardScaler

# Load the saved model
model = joblib.load("house_price_model.pkl")

print("Model loaded successfully!")



# New house data
new_house = pd.DataFrame([[
    5.0,
    20.0,
    6.0,
    1.0,
    1000.0,
    2.5,
    34.02,
    -118.25
]], columns=[
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
])

# Predict
prediction = model.predict(new_house)

print("Predicted house value:", prediction[0])
print(f"Predicted house value: ${prediction[0] * 100000:,.2f}")