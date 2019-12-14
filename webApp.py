import streamlit as st
import numpy as np
import pandas as pd
from math import floor
import dataHandler as dh

# Cached function from dataHandler
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


@st.cache
def doPCA(X):
    return dh.doPCA(X)


# Sidebar config
st.sidebar.header("ML project")
st.sidebar.markdown("By Mark Shapiro")
mode = st.sidebar.selectbox("Options", ["Home", "Preprocessing", "Model Selection"])

# Homepage
if mode == "Home":
    st.title("This is my homepage")
    st.image(
        "/home/shapim/Downloads/Shapiro_Mark.jpg", caption="Kennedy thinks I'm cute"
    )

# Visual stuff
elif mode == "Preprocessing":
    visual = st.sidebar.selectbox("See the data", ["--", "Images", "PCA"], 0)

    if visual == "--":
        st.title("Let's commit taboo and do data snooping")
        st.write('Choose "Images" to look at the digits')
        st.write('Or choose "PCA" to select principal components')

    elif visual == "Images":
        st.title("Look at pictures of the data")

        # Selection criteria
        ncol = st.number_input("Number of columns", 1, 10)
        nrow = st.number_input("Number of rows", 1, floor(50 / ncol))
        number = st.selectbox("Pick a number", ["--"] + [str(x) for x in range(10)], 0)

        # Retrieve the data and display images
        if number != "--":
            number = int(number)
            digits = splitData()
            img = dh.getPictures(digits, number, ncol, nrow)
            st.write(img)
            st.write("{} pictures of {} data".format(ncol * nrow, number))
    else:
        st.title("Principal Components")

        # Pick a number and see how good the approximation is
        numComponents = st.number_input(
            "Pick a number of components 1-256",
            min_value=0,
            max_value=256,
            value=0,
        )
        if numComponents > 0:
            # Get the data
            data, _, _, _ = getData()
            pca = doPCA(data)
            variances = pd.Series(pca.explained_variance_ratio_[:numComponents])

            # Plot 1
            st.subheader("Individual Explained Variances")
            st.line_chart(variances.rename("{0:.3f}%-{1:.3f}%".format(max(variances)*100, min(variances)*100)))

            # Plot 2
            cumulativeVariance = np.cumsum(variances)
            st.subheader("Sum of variances")
            st.line_chart(
                cumulativeVariance.rename("Explains {0:.3f}%".format(cumulativeVariance[len(cumulativeVariance) - 1] * 100))
            )

        # Pick a percentage and look for appropriate number of components
        st.header("and/or")

        desiredPercent = (
            st.number_input(
                "Pick a desired variance percent",
                min_value=0.00,
                max_value=100.00,
                value=0.00
            )/100
        )
        if desiredPercent > 0.00:
            # Get the data
            data, _, _, _ = getData()
            pca = doPCA(data)

            # Calculate and show closest match
            diff = [abs(desiredPercent-x) for x in np.cumsum(pca.explained_variance_ratio_)]
            closestNumber = diff.index(min(diff)) + 1
            st.write(str(closestNumber), "components needed for closest match")

# Solving with models
else:
    st.title("Model Selection")
    method = st.sidebar.selectbox(
        "Models all from scipy",
        ["--", "KNN", "SVM", "Linear", "NeuralNetwork", "Bayesian"]
    )

    if method == "--":
        st.write("Select a model")

    else:
        st.title(method)
        st.write("More to come")
