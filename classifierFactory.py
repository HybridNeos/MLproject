import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from random import sample
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical, plot_model

# TURN THIS INTO A FACTORY CLASS
class ClassifierFactory:
    # Plot decision boundary
    @staticmethod
    def plotBoundaries(X, y, clf, markers=None, colors=None):
        fig, ax = plt.subplots(1)
        ax = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        ax.set_yticklabels([])
        ax.tick_params(bottom="off", left="off")
        ax.set_xticklabels([])
        return fig

    # Scipy Classifiers
    @staticmethod
    def KNN(X, y, params):
        neigh = KNeighborsClassifier(params["K"])
        neigh.fit(X, y)
        return neigh

    @staticmethod
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

    # Neural Network Helpers
    @staticmethod
    def preview(model):
        # Save image of model
        filename = "KerasModel.png"
        plot_model(model, filename, True, True)

        #FAnders on stackoverflow for text view
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        return filename, short_model_summary

    @staticmethod
    def compile(params):
        model = Sequential()
        for i in range(len(params["numUnits"])):
            model.add(
                Dense(
                    units=params["numUnits"][i],
                    use_bias=True,
                    input_dim=2 if i == 0 else None,
                    activation=params["activations"][i],
                )
            )
        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["categorical_accuracy"])
        return model

    # Neural Network generator
    @staticmethod
    def NeuralNetwork(X, y, params):
        cat_y = to_categorical(y)
        model = ClassifierFactory.compile(params)
        for i in range(params["epochs"]):
            model.train_on_batch(X, cat_y)
        return model
