from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


def KNN(X, y, n):
    neigh = KNeighborsClassifier(n)
    neigh.fit(X, y)
    return neigh


def plotBoundaries(X, y, clf, markers=None, colors=None):
    fig, ax = plt.subplots(1)
    ax = (
        plot_decision_regions(X=X, y=y, clf=clf)
        if markers is None
        else plot_decision_regions(X=X, y=y, clf=clf, markers=markers, colors=colors)
    )
    ax.set_yticklabels([])
    ax.tick_params(bottom="off", left="off")
    ax.set_xticklabels([])
    return fig
