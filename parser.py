import numpy as np

def readData():
    train_data = np.loadtxt("./data/ZipDigits.train", dtype=float)
    test_data = np.loadtxt("./data/ZipDigits.test", dtype=float)

    y_train = train_data[:, 0].reshape(-1, 1).astype(int)
    X_train = train_data[:, 1:]

    y_test = test_data[:, 0].reshape(-1, 1).astype(int)
    X_test = test_data[:, 1:]

    return X_test, X_train, y_test, y_train

if __name__ == "__main__":
    readData()