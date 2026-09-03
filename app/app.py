from pathlib import Path
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_xgb_pipeline.pkl"

model = joblib.load(MODEL_PATH)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to estimate its price.")

st.divider()

st.subheader("🏡 House Details")

# --------------------------------------------------
# Row 1
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=7,
        help="Overall material and finish quality of the house."
    )

with col2:
    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=5000,
        value=1500,
        step=50
    )

# --------------------------------------------------
# Row 2
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    garage_cars = st.number_input(
        "Garage Capacity",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

with col2:
    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=1500,
        value=500,
        step=25
    )

# --------------------------------------------------
# Row 3
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2000,
        step=1
    )

with col2:
    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

# --------------------------------------------------
# Row 4
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    bedroom_abv_gr = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )

with col2:
    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=3000,
        value=1000,
        step=50
    )

# --------------------------------------------------
# Row 5
# --------------------------------------------------

lot_area = st.number_input(
    "Lot Area (sq ft)",
    min_value=500,
    max_value=100000,
    value=8000,
    step=500
)

st.divider()

if st.button("💰 Predict House Price"):

    # ----------------------------------------------
    # Input Validation
    # ----------------------------------------------

    if garage_cars > 0 and garage_area == 0:
        st.error(
            "Garage capacity is greater than 0, "
            "but garage area is 0."
        )
        st.stop()

    if bedroom_abv_gr > 0 and gr_liv_area < 300:
        st.error(
            "Living area is too small for the "
            "number of bedrooms entered."
        )
        st.stop()

    # ----------------------------------------------
    # Load Training Data as Template
    # ----------------------------------------------

    template_path = BASE_DIR / "models" / "house_input_template.pkl"

    template = joblib.load(template_path)

    house = template.copy()
    # ----------------------------------------------
    # Replace Template Values
    # ----------------------------------------------

    house["OverallQual"] = overall_qual
    house["GrLivArea"] = gr_liv_area
    house["GarageCars"] = garage_cars
    house["GarageArea"] = garage_area
    house["YearBuilt"] = year_built
    house["FullBath"] = full_bath
    house["BedroomAbvGr"] = bedroom_abv_gr
    house["TotalBsmtSF"] = total_bsmt_sf
    house["LotArea"] = lot_area

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    try:

        prediction = model.predict(house)[0]

        st.divider()

        st.subheader("💰 Estimated House Price")

        st.metric(
            label="Predicted Price",
            value=f"${prediction:,.2f}"
        )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)