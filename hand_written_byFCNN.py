import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_mnist(local_folder='mnist_data'):
    data_file = os.path.join(local_folder, 'mnist_data.joblib')
    X, y = joblib.load(data_file)
    return X, y


def split_dataset(X, y):
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
    return X_train, X_test, y_train, y_test


def train_nn(X_train, y_train, learning_rate, hidden_layer_sizes):
    # 训练全连接神经网络模型
    nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                     learning_rate_init=learning_rate,
                     random_state=42)
    nn.fit(X_train, y_train)
    return nn


def evaluate_nn(nn, X_test, y_test):
    # 评估神经网络模型
    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_count = (y_pred!= y_test).sum()
    return accuracy, error_count


def main():
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hidden_layer_sizes_list = [(500,), (1000,), (1500,), (2000,)]
    results = []
    for learning_rate in learning_rates:
        for hidden_layer_sizes in hidden_layer_sizes_list:
            nn = train_nn(X_train, y_train, learning_rate, hidden_layer_sizes)
            accuracy, error_count = evaluate_nn(nn, X_test, y_test)
            results.append((learning_rate, hidden_layer_sizes, accuracy, error_count))
            print(f"Learning Rate: {learning_rate}, Hidden Layer Sizes: {hidden_layer_sizes}, Accuracy: {accuracy}, Error Count: {error_count}")


if __name__ == "__main__":
    main()