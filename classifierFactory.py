import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from random import sample

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# TURN THIS INTO A FACTORY CLASS

def KNN(X, y, params):
    neigh = KNeighborsClassifier(params["K"])
    neigh.fit(X, y)
    return neigh


def SVM(X, y, params):
    # Linear
    if len(params) == 2:
        svm = SVC(params["C"], params["kernel"])
    # Sigmoid or rbf
    elif len(params) == 3:
        svm = SVC(params["C"], params["kernel"], 0, params["gamma"])
    # Poly
    elif len(params) == 4:
        svm = SVC(params["C"], params["kernel"], params["degree"], params["gamma"])

    svm.fit(X, y)
    return svm


def NeuralNetwork(X, y, params):
    cat_y = to_categorical(y)
    raise Exception("Not implemented yet")


def plotBoundaries(X, y, clf, markers=None, colors=None):
    fig, ax = plt.subplots(1)
    ax = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    ax.set_yticklabels([])
    ax.tick_params(bottom="off", left="off")
    ax.set_xticklabels([])
    return fig
