from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

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

def plotBoundaries(X, y, clf, markers=None, colors=None):
    fig, ax = plt.subplots(1)
    ax = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    ax.set_yticklabels([])
    ax.tick_params(bottom="off", left="off")
    ax.set_xticklabels([])
    return fig
