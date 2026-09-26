# 🏠 House Price Predictor for California Housing Dataset

A machine learning project that predicts California house values using the California Housing dataset and a Random Forest Regression model.

The project also includes an interactive Streamlit web application for making house-price predictions.

---

## 📌 Project Overview

House prices depend on several factors such as:

- Median income
- House age
- Average number of rooms
- Average number of bedrooms
- Population
- Average occupancy
- Latitude
- Longitude

This project uses these features to train a machine learning model that predicts the estimated median house value.

---

## 🎯 Objective

The main objectives of this project are:

- Understand a real-world regression problem
- Perform data exploration and preprocessing
- Train multiple machine learning models
- Compare model performance
- Tune the Random Forest model
- Save and load the trained model
- Build an interactive Streamlit application

---

## 📊 Dataset

The project uses the **California Housing Dataset**.

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Block population |
| `AveOccup` | Average house occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

### Target

`MedHouseVal` — Median house value.

The target is represented in units of **$100,000**.

---

## 🤖 Machine Learning Models

During the project, different regression approaches were explored:

### Linear Regression

Performance:

- MAE: **0.5332**
- R² Score: **0.5758**

### K-Nearest Neighbors Regression

Performance:

- MAE: **0.8128**
- R² Score: **0.1463**

### Random Forest Regression

Performance:

- MAE: **0.3267**
- R² Score: **0.8063**

The Random Forest model was selected for the final application based on its performance on the test dataset.

---

## ⚙️ Model Configuration

The final Random Forest model uses:

```text
n_estimators = 300
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
random_state = 42