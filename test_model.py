import pandas as pd
import joblib

# Load the saved model
model = joblib.load("final_house_price_model.pkl")

print("House Price Prediction")
print("----------------------")

# Get input from user
MedInc = float(input("Enter Median Income: "))
HouseAge = float(input("Enter House Age: "))
AveRooms = float(input("Enter Average Rooms: "))
AveBedrms = float(input("Enter Average Bedrooms: "))
Population = float(input("Enter Population: "))
AveOccup = float(input("Enter Average Occupancy: "))
Latitude = float(input("Enter Latitude: "))
Longitude = float(input("Enter Longitude: "))

# Create DataFrame
new_house = pd.DataFrame([[
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude
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

# Make prediction
prediction = model.predict(new_house)

# Display result
price = prediction[0] * 100000

print("----------------------")
print(f"Predicted House Value: ${price:,.2f}")