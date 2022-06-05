import pickle as pkl
import streamlit as st
import pandas as pd
import numpy as np


st.title('Breast Cancer Risk Prediction Application')

st.header('Machine Learning Application')

radius = st.number_input(label="Cell Radius", step=1., format="%.2f")
texture = st.number_input(label="Cell Texture", step=1., format="%.2f")
perimeter = st.number_input(label="Cell Perimeter", step=1., format="%.2f")
area = st.number_input(label="Cell Area", step=1., format="%.2f")
smoothness = st.number_input(label="Smoothness", step=1., format="%.2f")
compactness = st.number_input(label="Compactness", step=1., format="%.2f")
concavity = st.number_input(label="Concavity", step=1., format="%.2f")
concave_points = st.number_input(
    label="Concave Points", step=1., format="%.2f")
symmetry = st.number_input(label="Symmetry", step=1., format="%.2f")
fractal_dim = st.number_input(
    label="Fractal Dimensions", step=1., format="%.2f")

if st.button("Check Type!"):
    with open('pca10.pkl', 'rb') as f1:
        pca = pkl.load(f1)
    with open('rfc10.pkl', 'rb') as f2:
        rfc = pkl.load(f2)

    df = pd.DataFrame([[radius, texture, perimeter, area, smoothness,
                      compactness, concavity, concave_points, symmetry, fractal_dim]])
    sc_d = pca.transform(df)
    pred = rfc.predict(sc_d)
    if(pred == [1]):
        st.text(f"Prediction: MALIGNANT")
    else:
        st.text(f"Prediction: BENIGN")
