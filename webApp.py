import streamlit as st
import numpy as np
import pandas as pd
from math import floor
import digitHandler as dh

@st.cache
def getData():
    return dh.readData()

@st.cache
def splitData(trainOrTest="train"):
    if trainOrTest.lower() == "train":
        X_train, _, y_train, _ = getData()
        return dh.splitData(X_train, y_train)
    elif trainOrTest.lower() == "test":
        _, X_test, _, y_test = getData()
        return dh.splitData(X_test, y_test)
    else:
        raise Exception("Can only ask for train or test")

# Sidebar config
st.sidebar.header("ML project")
st.sidebar.markdown("By Mark Shapiro")

mode = st.sidebar.selectbox("Options", ["Home", "Visualization", "Model Selection"])

if mode == "Home":
    st.title("This is my homepage")
    st.image("/home/shapim/Downloads/Shapiro_Mark.jpg", caption="Kennedy thinks I'm cute")

elif mode == "Visualization":
    st.title("Provided by matplotlib")
    visual = st.sidebar.selectbox("See the data", ["--", "Images", "Plot"], 0)

    if visual == "--":
        st.write("Select a mode")

    elif visual == "Images":
        # Selection criteria
        ncol = st.number_input("Number of columns", 1, 10)
        nrow = st.number_input("Number of rows", 1, floor(50/ncol))
        number = st.selectbox("Pick a number", ["--"]+[str(x) for x in range(10)], 0)

        # Retrieve the data and display images
        if number != "--":
            number = int(number)
            digits = splitData()
            img = dh.imageTest(digits, number, ncol, nrow)
            st.image(img,caption="{} pictures of {} data".format(ncol*nrow,number))

    else:
        st.write("Plot goes here")

else:
    st.title("Model Selection")
    method = st.sidebar.selectbox("Models all from scipy", ["--", "KNN", "SVM", "Linear", "NeuralNetwork"])

    if method == "--":
        st.write("Select a model")

    else:
        st.title(method)
