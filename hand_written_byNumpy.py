import numpy as np
import gzip
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """
    从 MNIST 的 gz 文件中加载数据
    :param path: 文件所在路径
    :param kind: 数据类型
    :return: 图像数据和标签数据
    """
    labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
    images_path = f'{path}/{kind}-images-idx3-ubyte.gz'
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels


def relu(x):
    """
    ReLU 激活函数
    :param x: 输入
    :return: ReLU 计算结果
    """
    return np.maximum(0, x)


def softmax(x):
    """
    Softmax 函数，将输出转换为概率分布
    :param x: 输入
    :return: softmax 计算结果
    """
    # 对输入进行归一化处理，避免 exp 溢出
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    """
    交叉熵损失函数
    :param y_true: 真实标签
    :param y_pred: 预测概率
    :return: 交叉熵损失
    """
    m = y_true.shape[0]
    # 避免 np.log(0) 问题，给 y_pred 加上一个小的正数
    loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / m
    return loss


def one_hot_encode(labels, num_classes=10):
    """
    对标签进行独热编码
    :param labels: 原始标签
    :param num_classes: 类别数
    :return: 独热编码后的标签
    """
    m = labels.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), labels] = 1
    return one_hot


def forward_propagation(X, W1, b1, W2, b2):
    """
    前向传播过程
    :param X: 输入数据
    :param W1: 第一层权重
    :param b1: 第一层偏置
    :param W2: 第二层权重
    :param b2: 第二层偏置
    :return: 输出层的输出，隐藏层的输出
    """
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a2, a1


def backward_propagation(X, y, a2, a1, W2):
    """
    反向传播过程
    :param X: 输入数据
    :param y: 真实标签
    :param a2: 输出层的输出
    :param a1: 隐藏层的输出
    :param W2: 第二层权重
    :return: 权重和偏置的梯度
    """
    m = X.shape[0]
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0) / m
    dz1 = np.dot(dz2, W2.T)
    dz1[a1 <= 0] = 0  # 根据 ReLU 的导数修改 dz1
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0) / m
    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    """
    更新参数
    :param W1: 第一层权重
    :param b1: 第一层偏置
    :param W2: 第二层权重
    :param b2: 第二层偏置
    :param dW1: 第一层权重梯度
    :param db1: 第一层偏置梯度
    :param dW2: 第二层权重梯度
    :param db2: 第二层偏置梯度
    :param learning_rate: 学习率
    :return: 更新后的权重和偏置
    """
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


def train(X_train, y_train, hidden_size=128, learning_rate=0.01, epochs=100):
    """
    训练神经网络
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param hidden_size: 隐藏层节点数
    :param learning_rate: 学习率
    :param epochs: 训练轮数
    :return: 训练好的权重和偏置
    """
    input_size = X_train.shape[1]
    output_size = 10
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)
    for epoch in range(epochs):
        a2, a1 = forward_propagation(X_train, W1, b1, W2, b2)
        loss = cross_entropy_loss(y_train, a2)
        dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, a2, a1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return W1, b1, W2, b2


def predict(X_test, W1, b1, W2, b2):
    """
    预测函数
    :param X_test: 测试数据
    :param W1: 第一层权重
    :param b1: 第一层偏置
    :param W2: 第二层权重
    :param b2: 第二层偏置
    :return: 预测结果
    """
    a2, _ = forward_propagation(X_test, W1, b1, W2, b2)
    predictions = np.argmax(a2, axis=1)
    return predictions


def evaluate(X_test, y_test, W1, b1, W2, b2):
    """
    评估函数，计算准确率和错误数量
    :param X_test: 测试数据
    :param y_test: 测试标签
    :param W1: 第一层权重
    :param b1: 第一层偏置
    :param W2: 第二层权重
    :param b2: 第二层偏置
    :return: 准确率，错误数量
    """
    predictions = predict(X_test, W1, b1, W2, b2)
    correct = np.sum(predictions == y_test)
    accuracy = correct / y_test.shape[0]
    error_count = y_test.shape[0] - correct
    return accuracy, error_count


def main():
    X_train, y_train = load_mnist('MNIST', kind='train')
    X_test, y_test = load_mnist('MNIST', kind='t10k')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = one_hot_encode(y_train)
    hidden_sizes = [500, 1000, 1500, 2000]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    results = []
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            print(f'Training with hidden size {hidden_size} and learning rate {learning_rate}')
            W1, b1, W2, b2 = train(X_train, y_train, hidden_size, learning_rate)
            accuracy, error_count = evaluate(X_test, y_test, W1, b1, W2, b2)
            results.append((hidden_size, learning_rate, accuracy, error_count))
    print("Experiment Results:")
    for result in results:
        print(f'Hidden Size: {result[0]}, Learning Rate: {result[1]}, Accuracy: {result[2]}, Error Count: {result[3]}')


if __name__ == "__main__":
    main()