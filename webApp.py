import streamlit as st
import numpy as np
import pandas as pd
from math import floor
import dataHandler as dh
import classifierFactory as cf
from random import sample


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


# @st.cache
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


# @st.cache
def projectOnto(data, numComponents):
    pca = doPCA(getData()[0])
    return data @ pca.components_[:numComponents].T


# @st.cache
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


pointsPerDigit = 10


def pickNumbers(model):
    st.header("Pick digits to show")
    return st.multiselect("0-9 available", range(10), key=model)


def paramStart():
    st.write("")
    st.header("Parameters")
    return {}


def plotBoundaries(numbers, model, params):
    st.header("\nDecision boundary")

    # Get the data
    subset_X, subset_y = sizeAndDigitSubset(numbers, pointsPerDigit)
    reduced = projectOnto(subset_X, 2)

    # function mapping
    models = {"KNN": cf.KNN, "SVM": cf.SVM}

    # Complete the plot
    if st.button("Plot it"):
        with st.spinner("Constructing classifer"):
            clf = models[model](reduced, subset_y, params)
            Ein = 1 - clf.score(reduced, subset_y)

        with st.spinner("Drawing boundaries"):
            # m, c = getColorsAndMarkers(numbers)
            img = cf.plotBoundaries(reduced, subset_y, clf)
            st.write(img)

        st.subheader("Classifcation error of {0:.3f}".format(Ein))


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
            totals = np.cumsum(pca.explained_variance_ratio_)
            diff = [abs(desiredPercent - x) for x in totals]
            closestNumber = diff.index(min(diff)) + 1
            st.write(
                "{0} gives closest match at {1:.3f}% variance".format(
                    str(closestNumber), 100*totals[closestNumber - 1]
                )
            )

# Preview images of models
elif mode == "Model Preview":
    model = st.sidebar.selectbox(
        "Choose a model to preview its hypothesis boundary",
        [
            "--",
            "KNN",
            "SVM",
            "NeuralNetwork",
            "Regression",
            "Bayesian",
            "Random Forest",
        ],
    )

    if model == "--":
        st.header("Select a model")
        st.write("Data drawn is randomly sampled each time but cached")
        st.write("The mlxtend plot_decision_regions functions is used")
        st.write("--It is buggy")

    else:
        st.title(model)

        # Get a portion of the data
        if model == "KNN":
            # Numbers
            numbers = pickNumbers(model)
            maxK = max(1, pointsPerDigit * len(numbers))

            # Parameters
            params = paramStart()
            params["K"] = st.number_input("Pick K from 1 to " + str(maxK), 1, maxK)

            # Plot
            plotBoundaries(numbers, model, params)

        elif model == "SVM":
            # Numbers
            numbers = pickNumbers(model)

            # Parameters
            params = paramStart()
            params["C"] = st.number_input(
                "Regularization paramater C", min_value=0.01, value=1.0
            )
            params["kernel"] = st.selectbox(
                "Kernel", ["rbf", "linear", "poly", "sigmoid"]
            )
            if params["kernel"] == "poly":
                params["degree"] = st.number_input("Polynomial order", 1, value=3)
            if params["kernel"] != "linear":
                gamma = st.text_input(
                    "Kernel coefficient. Enter 'scale', 'auto', or a float", "scale"
                )
                params["gamma"] = (
                    float(gamma) if gamma.replace(".", "").isnumeric() else gamma
                )

            # Plot
            plotBoundaries(numbers, model, params)

        elif model == "NeuralNetwork":
            # Numbers
            numbers = pickNumbers(model)

            # Parameters
            params = paramStart()
            numLayers = st.number_input("Number of hidden layers", 1, 5)
            params["numUnits"] = ["None" for _ in range(numLayers+1)]
            params["activation"] = [0 for _ in range(numLayers+1)]

            # Hidden Layers
            st.subheader("Hidden Layers")
            for i in range(numLayers):
                st.write("Layer "+str(i+1))
                params["numUnits"][i] = st.number_input("Number of units", 1, 64, 16, key=i)
                params["activation"][i] = st.selectbox(
                    "Activation function",
                    ["elu", "softmax", "selu", "softplus", "softsign", "relu",
                    "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear"],
                    6 if i < numLayers-1 else 10,
                    key=i
                )
                if i < numLayers-1:
                    st.write("")

            st.subheader("Output layer")
            params["numUnits"][-1] = 1 if len(numbers) == 2 else len(numbers)
            params["activation"][-1] = st.selectbox(
                "Output layer activation function",
                ["elu", "softmax", "selu", "softplus", "softsign", "relu",
                "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear"],
                6 if i < numLayers-1 else 10,
                key=i
            )

        else:
            st.title("I don't know how " + model + " works")
            st.write("Coming soon")

# Actual Solving
else:
    st.title("Actual solving will go here")
