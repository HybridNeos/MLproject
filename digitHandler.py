from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")


def readData():
    train_data = np.loadtxt("./Data/ZipDigits.train", dtype=float)
    test_data = np.loadtxt("./Data/ZipDigits.test", dtype=float)

    # normalize = lambda x: -x
    normalize = lambda x: (1 - (x + 1) / 2)
    # normalize = lambda x: x

    y_train = train_data[:, 0].reshape(-1, 1).astype(int)
    X_train = normalize(train_data[:, 1:])

    y_test = test_data[:, 0].reshape(-1, 1).astype(int)
    X_test = normalize(test_data[:, 1:])

    return X_train, X_test, y_train, y_test


def splitData(X, y):
    digits = [None for _ in range(10)]

    for i in range(10):
        indices = [True if y[j] == i else False for j in range(len(y))]
        digits[i] = X[indices]

    return digits


def imageTest(digits, offset=0, number, ncol, nrow):
    # error checking
    if ncol * nrow > 50:
        raise Exception("Too many images asked for")

    images = digits[number]

    # Make the plot
    if ncol == 1 and nrow == 1:
        digit = images[0].reshape(16, 16)
        plt.imshow(digit, cmap="gray")
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
    else:
        fig, ax = plt.subplots(ncol, nrow)
        #fig.tight_layout()
        for i in range(ncol):
            for j in range(nrow):
                unrolled = (i * nrow) + j
                digit = images[unrolled+offset].reshape(16, 16)
                ax[i, j].imshow(digit, cmap="gray")
                ax[i, j].set_yticklabels([])
                ax[i, j].set_xticklabels([])

    # Save the plot
    pipe = BytesIO()
    plt.savefig(pipe, format="png")

    # Read the saved figure into Image
    pipe.seek(0)
    img = Image.open(pipe)
    img.show()
    pipe.close()

    return None


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = readData()
    train_digits = splitData(X_train, y_train)
    test_digits = splitData(X_test, y_test)
    imageTest(train_digits, 0, 4, 4)
