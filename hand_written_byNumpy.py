import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

#实现sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#实现softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#加载MNIST数据集
def load_mnist(local_folder='mnist_data'):
    data_file = os.path.join(local_folder, 'mnist_data.joblib')
    X, y = joblib.load(data_file)
    return X, y

#对标签进行编码
def one_hot_encode(y, num_classes):
    num_samples = len(y)
    encoded = np.zeros((num_samples, num_classes))
    encoded[np.arange(num_samples), y.astype(int)] = 1
    return encoded


def split_dataset(X, y):
    # 分割数据集为训练集和测试集
    indices = np.random.permutation(len(X))
    train_size = 60000
    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    X_test = X[indices[train_size:]]
    y_test = y[indices[train_size:]]
    return X_train, X_test, y_train, y_test

#前向传播
def forward_propagation(X, weights, biases):
    a = X
    activations = [X]
    for i in range(len(weights) - 1):
        z = np.dot(a, weights[i]) + biases[i]
        a = sigmoid(z)
        activations.append(a)
    z = np.dot(a, weights[-1]) + biases[-1]
    a = softmax(z)
    activations.append(a)
    return activations

#反向传播
def back_propagation(X, y, activations, weights, biases, learning_rate):
    num_layers = len(weights)
    m = X.shape[0]
    dz = activations[-1] - y
    gradients = []
    for i in reversed(range(num_layers)):
        dw = np.dot(activations[i].T, dz) / m
        db = np.sum(dz, axis=0) / m
        gradients.append((dw, db))
        if i > 0:
            dz = np.dot(dz, weights[i].T) * (activations[i] * (1 - activations[i]))
    gradients.reverse()
    return gradients

#更新权重
def update_weights(weights, biases, gradients, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients[i][0]
        biases[i] -= learning_rate * gradients[i][1]
    return weights, biases

#训练神经网络
def train_nn(X_train, y_train, learning_rate, hidden_layer_sizes, num_epochs, num_classes):
    input_size = X_train.shape[1]
    output_size = num_classes
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) / np.sqrt(layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
    biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
    for epoch in range(num_epochs):
        activations = forward_propagation(X_train, weights, biases)
        gradients = back_propagation(X_train, y_train, activations, weights, biases, learning_rate)
        weights, biases = update_weights(weights, biases, gradients, learning_rate)
    return weights, biases

#评估神经网络
def evaluate_nn(X_test, y_test, weights, biases):
    activations = forward_propagation(X_test, weights, biases)
    y_pred = np.argmax(activations[-1], axis=1)
    accuracy = np.mean(y_pred == y_test)
    error_count = np.sum(y_pred!= y_test)
    return accuracy, error_count


def main():
    X, y = load_mnist()
    num_classes = 10
    y_train_encoded = one_hot_encode(y[:60000], num_classes)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hidden_layer_sizes_list = [[500], [1000], [1500], [2000]]
    num_epochs = 10
    results = []
    for learning_rate in learning_rates:
        for hidden_layer_sizes in hidden_layer_sizes_list:
            weights, biases = train_nn(X_train, y_train_encoded[:60000], learning_rate, hidden_layer_sizes, num_epochs, num_classes)
            accuracy, error_count = evaluate_nn(X_test, y_test[60000:], weights, biases)
            results.append((learning_rate, hidden_layer_sizes, accuracy, error_count))
            print(f"Learning Rate: {learning_rate}, Hidden Layer Sizes: {hidden_layer_sizes}, Accuracy: {accuracy}, Error Count: {error_count}")


if __name__ == "__main__":
    main()