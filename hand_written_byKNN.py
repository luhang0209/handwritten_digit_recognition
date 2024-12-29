import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import gzip
import struct


def load_mnist(path, kind='train'):
    """
    从文件中加载 MNIST 数据集
    :param path: 数据集文件的路径
    :param kind: 数据集类型，可选 'train' 或 'test'
    :return: 图像数据和标签数据的 numpy 数组
    """
    labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
    images_path = f'{path}/{kind}-images-idx3-ubyte.gz'
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def main():
    # 加载训练集和测试集
    X_train, y_train = load_mnist('MNIST', kind='train')
    X_test, y_test = load_mnist('MNIST', kind='t10k')
    
    k_values = [1, 3, 5, 7]  # 不同的 k 值
    accuracy_list = []
    error_count_list = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        error_count = len(y_test) - np.sum(y_pred == y_test)
        accuracy_list.append(accuracy)
        error_count_list.append(error_count)
        print(f"k = {k}: Accuracy = {accuracy:.4f}, Error Count = {error_count}")
    
    # 绘制表格
    fig, ax = plt.subplots()
    cell_text = []
    for i in range(len(k_values)):
        cell_text.append([k_values[i], accuracy_list[i], error_count_list[i]])
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=cell_text, colLabels=['k value', 'Accuracy', 'Error Count'], loc='center')
    plt.show()


if __name__ == "__main__":
    main()