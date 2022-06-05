import pickle as pkl
import streamlit as st
import pandas as pd
import numpy as np


st.title('Breast Cancer Risk Prediction Application')

st.header(
    'Machine Learning Application to Predict the Cancer Class - Benign | Malignant')
st.write("Enter the details from your FNA test for a prediction: ")
radius = st.number_input(label="Cell Radius", step=1., format="%.2f")
texture = st.number_input(label="Cell Texture", step=1., format="%.2f")
perimeter = st.number_input(label="Cell Perimeter", step=1., format="%.2f")
area = st.number_input(label="Cell Area", step=1., format="%.2f")
smoothness = st.number_input(label="Smoothness", step=1., format="%.4f")
compactness = st.number_input(label="Compactness", step=1., format="%.4f")
concavity = st.number_input(label="Concavity", step=1., format="%.4f")
concave_points = st.number_input(
    label="Concave Points", step=1., format="%.4f")
symmetry = st.number_input(label="Symmetry", step=1., format="%.4f")
fractal_dim = st.number_input(
    label="Fractal Dimensions", step=1., format="%.4f")

if st.button("Submit."):
    # with open('pca (2).pkl', 'rb') as f1:
    #     pca = pkl.load(f1)
    df = pd.DataFrame([[radius, texture, perimeter, area, smoothness,
                      compactness, concavity, concave_points, symmetry, fractal_dim]])
    if ((df.values == 0).all()):
        st.error("Enter valid details!")
    else:
        with open('rfc-np10.pkl', 'rb') as f2:
            rfc = pkl.load(f2)

        # sc_d = pca.transform(df)
        # pred = rfc.predict(sc_d)
        pred = rfc.predict(df)

        st.success('Prediction computed!')
        if(pred == [1]):
            st.text(f"Prediction: MALIGNANT")
        else:
            st.text(f"Prediction: BENIGN")

with st.expander("Learn More"):
    st.write("Developed by Roshni Balasubramanian")
    st.write("2019115083, IT Department, Anna University")
    st.write("Socially Relevant Project - Semester VI")
    st.write(" ")
    st.write("The purpose of this project is to learn and implement Machine Learning. The best model identified was the Random Forest Classifier.")
    st.write(
        "GitHub Repository: https://github.com/Roshni-Bala/Breast-Cancer-Risk-Prediction")
