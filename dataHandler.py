from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("agg")

# Get the bases data
def readData():
    train_data = np.loadtxt("./Data/ZipDigits.train", dtype=float)
    test_data = np.loadtxt("./Data/ZipDigits.test", dtype=float)

    normalize = lambda x: (1 - (x + 1) / 2)

    y_train = train_data[:, 0].reshape(-1,).astype(int)
    X_train = normalize(train_data[:, 1:])

    y_test = test_data[:, 0].reshape(-1,).astype(int)
    X_test = normalize(test_data[:, 1:])

    return X_train, X_test, y_train, y_test

# Split the data by digit
def splitData(X, y, numbers=None):
    digits = [None for _ in range(10)]

    for i in range(10):
        indices = [True if y[j] == i else False for j in range(len(y))]
        digits[i] = X[indices]

    return digits

# Get a subset of the digits
def selectDigits(X, y, numbers):
    #for i in numbers:
        #stuff
    return None

def getPictures(digits, number, ncol, nrow, offset=0):
    # error checking
    if ncol * nrow > 50:
        raise Exception("Too many images asked for")

    images = digits[number]

    # Make the plot
    if ncol == 1 and nrow == 1:
        digit = images[0].reshape(16, 16)
        fig, ax = plt.subplots(1)
        ax.imshow(digit, cmap="gray")
        ax.set_yticklabels([])
        ax.tick_params(bottom='off', left='off')
        ax.set_xticklabels([])
    elif ncol == 1 or nrow == 1:
        fig, ax = plt.subplots(ncol, nrow)
        for i in range(nrow if ncol == 1 else ncol):
            digit = images[i+offset].reshape(16, 16)
            ax[i].imshow(digit, cmap="gray")
            ax[i].set_yticklabels([])
            ax[i].tick_params(bottom='off', left='off')
            ax[i].set_xticklabels([])
    else:
        fig, ax = plt.subplots(ncol, nrow)
        #fig.tight_layout()
        for i in range(ncol):
            for j in range(nrow):
                unrolled = (i * nrow) + j
                digit = images[unrolled+offset].reshape(16, 16)
                ax[i, j].imshow(digit, cmap="gray")
                ax[i, j].set_yticklabels([])
                ax[i, j].tick_params(bottom='off', left='off')
                ax[i, j].set_xticklabels([])

    return fig

def doPCA(X):
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    return pca

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = readData()
    #train_digits = splitData(X_train, y_train)
    #test_digits = splitData(X_test, y_test)
    #imageTest(train_digits, 0, 4, 4)
