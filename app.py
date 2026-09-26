import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load("final_house_price_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏠 House Price Predictor")

st.write(
    "Enter the house details below to predict its estimated value "
    "using a trained Random Forest model."
)

st.divider()


# --------------------------------------------------
# House Details
# --------------------------------------------------

st.subheader("📋 House Details")

col1, col2 = st.columns(2)

with col1:

    MedInc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=5.0
    )

    HouseAge = st.number_input(
        "House Age",
        min_value=0.0,
        value=20.0
    )

    AveRooms = st.number_input(
        "Average Rooms",
        min_value=0.0,
        value=6.0
    )

    AveBedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.0,
        value=1.0
    )


with col2:

    Population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0
    )

    AveOccup = st.number_input(
        "Average Occupancy",
        min_value=0.0,
        value=2.5
    )

    Latitude = st.number_input(
        "Latitude",
        value=34.02
    )

    Longitude = st.number_input(
        "Longitude",
        value=-118.25
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict House Price", use_container_width=True):

    # -------------------------------
    # Input validation
    # -------------------------------

    if AveRooms < AveBedrms:

        st.error(
            "❌ Average Rooms should be greater than "
            "Average Bedrooms."
        )

    elif AveOccup <= 0:

        st.error(
            "❌ Average Occupancy must be greater than 0."
        )

    elif Population <= 0:

        st.error(
            "❌ Population must be greater than 0."
        )

    else:

        # -------------------------------
        # Create input DataFrame
        # -------------------------------

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


        # -------------------------------
        # Make prediction
        # -------------------------------

        prediction = model.predict(new_house)

        price = prediction[0] * 100000


        # -------------------------------
        # Display prediction
        # -------------------------------

        st.success("✅ Prediction completed successfully!")

        st.metric(
            label="🏠 Estimated House Value",
            value=f"${price:,.2f}"
        )

        st.info(
            "This prediction is generated using the trained "
            "Random Forest model."
        )


# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "R² Score",
        "0.8063"
    )

with col2:

    st.metric(
        "MAE",
        "0.3267"
    )

with col3:

    st.metric(
        "Trees",
        "300"
    )


st.caption(
    "Model: Random Forest Regressor | "
    "R² = 0.8063 | MAE = 0.3267"
)


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.divider()

st.subheader("📈 Feature Importance")

feature_names = [
    "Median Income",
    "House Age",
    "Average Rooms",
    "Average Bedrooms",
    "Population",
    "Average Occupancy",
    "Latitude",
    "Longitude"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature"),
    horizontal=True
)

st.caption(
    "Higher values indicate that the feature was more important "
    "to the Random Forest model when making predictions."
)