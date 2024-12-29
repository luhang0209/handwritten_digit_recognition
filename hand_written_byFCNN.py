import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_mnist(path, kind='train'):
    """
    该函数用于加载 MNIST 数据集。
    :param path: 数据集所在的路径。
    :param kind: 要加载的数据集类型
    :return: 图像数据和标签数据。
    """
    # 构造标签文件的路径
    labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
    # 构造图像文件的路径
    images_path = f'{path}/{kind}-images-idx3-ubyte.gz'
    with gzip.open(labels_path, 'rb') as lbpath:
        # 从标签文件中读取数据，数据类型为无符号 8 位整数，偏移量为 8 字节开始读取
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        # 从图像文件中读取数据，数据类型为无符号 8 位整数，偏移量为 16 字节开始读取，并重塑为 (样本数量, 784) 的形状
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels


def main():
    # 调用 load_mnist 函数加载训练集数据
    X_train, y_train = load_mnist('MNIST', kind='train')
    # 调用 load_mnist 函数加载测试集数据
    X_test, y_test = load_mnist('MNIST', kind='t10k')

    # 数据归一化，将图像数据归一化到 [0, 1] 范围
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # 定义不同的学习率
    learning_rates = [0.1,0.01, 0.001, 0.0001]
    # 定义不同的隐层节点数组合
    hidden_layer_sizes = [(500,), (1000,), (1500,),(2000,)]
    results = []
    
    for lr in learning_rates:
        for hidden_size in hidden_layer_sizes:
            # 创建一个 MLPClassifier 模型，指定隐层节点数和初始学习率，最大迭代次数为 500
            model = MLPClassifier(hidden_layer_sizes=hidden_size, learning_rate_init=lr, max_iter=1000,random_state=5,alpha=0.0001,solver='sgd')
            # 使用训练集数据训练模型
            model.fit(X_train, y_train)
            
            # 使用训练好的模型对测试集进行预测
            y_pred = model.predict(X_test)
            
            # 计算预测的准确率，使用 accuracy_score 函数对比真实标签和预测标签
            accuracy = accuracy_score(y_test, y_pred)
            # 计算错误数量，通过测试集样本数量减去预测正确的样本数量
            error_count = len(y_test) - np.sum(y_test == y_pred)
            
            # 将本次实验结果存储在结果列表中
            results.append({
                'learning_rate': lr,
                'hidden_layer_size': hidden_size,
                'accuracy': accuracy,
                'error_count': error_count
            })
            
            # 打印本次实验结果
            print(f"Learning Rate: {lr}, Hidden Layer Size: {hidden_size}, Accuracy: {accuracy}, Error Count: {error_count}")
    
    for result in results:
        print(f"学习率: {result['learning_rate']}, 隐层节点数: {result['hidden_layer_size']}, 准确率: {result['accuracy']}, 错误数量: {result['error_count']}")


if __name__ == "__main__":
    main()