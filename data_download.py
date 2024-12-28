import os
import numpy as np
import joblib
from sklearn.datasets import fetch_openml


def download_mnist(local_folder='mnist_data'):
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    data_file = os.path.join(local_folder, 'mnist_data.joblib')
    if os.path.exists(data_file):
        print("Loading MNIST data from local file...")
        X, y = joblib.load(data_file)
    else:
        print("Downloading MNIST data from the internet...")
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data.astype('float32')
        y = mnist.target.astype('int')
        joblib.dump((X, y), data_file)
    return X, y


def main():
    X, y = download_mnist()
    print(f"Number of samples: {len(X)}")
    print(f"Number of labels: {len(y)}")


if __name__ == "__main__":
    main()