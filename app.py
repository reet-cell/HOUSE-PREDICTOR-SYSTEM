import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("pipeline.pkl", "rb"))

st.title("🏠 House Price Prediction")

area = st.slider("Living Area", 500, 5000, 1500)
bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5])
bathrooms = st.selectbox("Bathrooms", [1,2,3])
location = st.selectbox("Neighborhood", ["NAmes","CollgCr","OldTown","Edwards"])

if st.button("Predict"):
    input_df = pd.DataFrame({
        'GrLivArea': [area],
        'BedroomAbvGr': [bedrooms],
        'FullBath': [bathrooms],
        'Neighborhood': [location]
    })

    price = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${price:,.2f}")