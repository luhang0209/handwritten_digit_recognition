import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def load_mnist(local_folder='mnist_data'):
    data_file = os.path.join(local_folder, 'mnist_data.joblib')
    X, y = joblib.load(data_file)
    return X, y


def split_dataset(X, y):
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
    return X_train, X_test, y_train, y_test


def train_knn(X_train, y_train, k):
    # 训练 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


def evaluate_knn(knn, X_test, y_test):
    # 评估 KNN 模型
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # 计算错误数量
    error_count = (y_pred!= y_test).sum()
    return accuracy, error_count


def main():
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    k_values = [1, 3, 5, 7]
    results = []
    for k in k_values:
        knn = train_knn(X_train, y_train, k)
        accuracy, error_count = evaluate_knn(knn, X_test, y_test)
        results.append((accuracy, error_count))
        print(f"K = {k}, Accuracy: {accuracy}, Error Count: {error_count}")
    # 绘制准确率结果
    accuracies = [result[0] for result in results]
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('KNN Accuracy vs K value')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    # 绘制错误数量结果
    error_counts = [result[1] for result in results]
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, error_counts, marker='o', linestyle='-', color='r')
    plt.title('KNN Error Count vs K value')
    plt.xlabel('K value')
    plt.ylabel('Error Count')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()