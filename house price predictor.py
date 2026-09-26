import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score


housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.to_csv("data.csv", index=False)


X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 1. Initialize and train the model


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(X_train_scaled[:5])

# 3. Evaluate model performance


print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")
print(f"R² Score: {rf_r2:.4f}")

importance = rf_model.feature_importances_
print("Feature importances:")
for feature, value in zip(X.columns, importance):
    print(feature, ":", value)

MedInc = float(input("Enter median income: "))
HouseAge = float(input("Enter house age: "))
AveRooms = float(input("Enter average rooms: "))
AveBedrms = float(input("Enter average bedrooms: "))
Population = float(input("Enter population: "))
AveOccup = float(input("Enter average occupancy: "))
Latitude = float(input("Enter latitude: "))
Longitude = float(input("Enter longitude: "))

new_house = [[
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude
]]

prediction = rf_model.predict(new_house)

print("Predicted house value:", prediction[0])
print(f"Predicted house value: ${prediction[0] * 100000:.2f}")
print("New house data:", new_house[0])



joblib.dump(rf_model, "house_price_model.pkl")

print("Model saved successfully!")

new_house = [[5, 20, 6, 1, 1000, 2.5, 34.02, -118.25]]



knn_model = KNeighborsRegressor(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_mae = mean_absolute_error(y_test, knn_predictions)
knn_r2 = r2_score(y_test, knn_predictions)

print("KNN without scaling")
print("MAE:", knn_mae)
print("R²:", knn_r2)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

for n in [50, 100, 200, 300]:

    model = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Trees:", n)
    print("MAE:", mae)
    print("R²:", r2)
    print("----------------")
 
 
for split in [2, 5, 10, 20]:

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=split,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Min Samples Split:", split)
    print("MAE:", mae)
    print("R²:", r2)
    print("----------------")
    
    
for leaf in [1, 2, 4, 8]:

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=leaf,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Min Samples Leaf:", leaf)
    print("MAE:", mae)
    print("R²:", r2)
    print("----------------")
    
    
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

final_mae = mean_absolute_error(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)

print("Final Model Trained Successfully!")
print("Final MAE:", final_mae)
print("Final R²:", final_r2)

import joblib

joblib.dump(final_model, "final_house_price_model.pkl")

print("Final model saved successfully!")