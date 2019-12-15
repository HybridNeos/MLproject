import streamlit as st
import numpy as np
import pandas as pd
from math import floor
import dataHandler as dh
import classifiers as cf
from random import sample

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
def sizeAndDigitSubset(numbers, n=10):
    # Get data prepare output holder
    digits = splitData()
    X = np.empty((len(numbers) * n, digits[0].shape[1]))
    y = np.empty((len(numbers) * n,), dtype=int)

    # recombine
    for i, number in enumerate(numbers):
        X[i * n : (i + 1) * n] = sample(list(digits[number]), n)
        y[i * n : (i + 1) * n] = number

    return X, y


@st.cache
def doPCA(X):
    return dh.doPCA(X)


@st.cache
def projectOnto(data, numComponents):
    pca = doPCA(getData()[0])
    return data @ pca.components_[:numComponents].T


@st.cache
def getColorsAndMarkers(numbers):
    markers = ["s", "^", "o", "X", "v", "<", ">", "P", "h", "D"]
    colors = [
        "r",
        "b",
        "limegreen",
        "lightgray",
        "cyan",
        "m",
        "y",
        "darkorange",
        "chocolate",
        "tab:purple",
    ]
    m = [markers[num] for num in numbers]
    c = [colors[num] for num in numbers]
    return "".join(m), ",".join(c)


################################################################################

# Sidebar config
st.sidebar.header("ML project")
st.sidebar.markdown("By Mark Shapiro")
mode = st.sidebar.selectbox(
    "Options", ["Home", "Preprocessing", "Model Preview", "Solving"]
)

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
            "Pick a number of components 1-256", min_value=0, max_value=256, value=0
        )
        if numComponents > 0:
            # Get the data
            data, _, _, _ = getData()
            pca = doPCA(data)
            variances = pd.Series(pca.explained_variance_ratio_[:numComponents])

            # Plot 1
            st.subheader("Individual Explained Variances")
            st.line_chart(
                variances.rename(
                    "{0:.3f}%-{1:.3f}%".format(
                        max(variances) * 100, min(variances) * 100
                    )
                )
            )

            # Plot 2
            cumulativeVariance = np.cumsum(variances)
            st.subheader("Sum of variances")
            st.line_chart(
                cumulativeVariance.rename(
                    "Explains {0:.3f}%".format(
                        cumulativeVariance[len(cumulativeVariance) - 1] * 100
                    )
                )
            )

        # Pick a percentage and look for appropriate number of components
        st.header("and/or")

        desiredPercent = (
            st.number_input(
                "Pick a desired variance percent",
                min_value=0.00,
                max_value=100.00,
                value=0.00,
            )
            / 100
        )
        if desiredPercent > 0.00:
            # Get the data
            data, _, _, _ = getData()
            pca = doPCA(data)

            # Calculate and show closest match
            diff = [
                abs(desiredPercent - x)
                for x in np.cumsum(pca.explained_variance_ratio_)
            ]
            closestNumber = diff.index(min(diff)) + 1
            st.write(str(closestNumber), "components needed for closest match")

# Preview images of models
elif mode == "Model Preview":
    method = st.sidebar.selectbox(
        "Choose a model to preview its hypothesis boundary",
        ["--", "KNN", "SVM", "Linear", "NeuralNetwork", "Bayesian"],
    )

    if method == "--":
        st.header("Select a model")
        st.write("Data drawn is randomly sampled each time but cached")
        st.write("The mlxtend plot_decision_regions functions is used")
        st.write("--It is buggy")

    else:
        # How many data points per digit we have
        n = 10

        # Get a portion of the data
        if method == "KNN":
            st.title("K Nearest Neighbors")

            # Paramaters and data shrinking
            numbers = st.multiselect("Pick digits to show", np.arange(10))
            # numbers = [int(num) for num in range(10)]
            maxK = max(1, n * len(numbers))
            K = st.number_input("Pick K from 1 to " + str(maxK), 1, maxK)

            subset_X, subset_y = sizeAndDigitSubset(numbers, n)
            reduced = projectOnto(subset_X, 2)

            # Plot the decision boundaries
            st.write("")
            if st.button("Plot it"):
                with st.spinner("Constructing classifer"):
                    clf = cf.KNN(reduced, subset_y, K)
                    Ein = 1 - clf.score(reduced, subset_y)
                with st.spinner("Drawing boundaries"):
                    m, c = getColorsAndMarkers(numbers)
                    img = cf.plotBoundaries(reduced, subset_y, clf, markers=m, colors=c)
                    st.write(img)
                st.subheader("Classifcation error of " + str(Ein))

        else:
            st.title(method)
            st.write("More to come")

# Actual Solving
else:
    st.title("Actual solving will go here")
