import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self, hidden_units):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_mnist():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch, learning_rate):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    error_count = len(test_loader.dataset) - correct
    return accuracy, error_count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hidden_units_list = [500, 1000, 1500, 2000]
    results = []
    for learning_rate in learning_rates:
        for hidden_units in hidden_units_list:
            model = CNN(hidden_units).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            train_loader, test_loader = load_mnist()
            for epoch in range(5):
                train(model, device, train_loader, optimizer, epoch, learning_rate)
            accuracy, error_count = test(model, device, test_loader)
            results.append((learning_rate, hidden_units, accuracy, error_count))
            print(f"Learning Rate: {learning_rate}, Hidden Units: {hidden_units}, Accuracy: {accuracy}, Error Count: {error_count}")
    # 绘制结果
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    for i, learning_rate in enumerate(learning_rates):
        accuracies = []
        error_counts = []
        for result in results:
            if result[0] == learning_rate:
                accuracies.append(result[2])
                error_counts.append(result[3])
        axs[i].plot([str(units) for units in hidden_units_list], accuracies, marker='o', label='Accuracy')
        axs[i].plot([str(units) for units in hidden_units_list], error_counts, marker='x', label='Error Count')
        axs[i].set_title(f'Learning Rate = {learning_rate}')
        axs[i].set_xlabel('Hidden Units')
        axs[i].set_ylabel('Value')
        axs[i].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()