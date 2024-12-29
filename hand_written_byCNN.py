import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, hidden_nodes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    return trainloader, testloader


def train_and_evaluate(model, trainloader, testloader, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)  # 将输入数据移动到 GPU
            labels = labels.to(device)  # 将标签数据移动到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {running_loss / len(trainloader)}")

    correct = 0
    total = 0
    error_count = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)  # 将测试数据移动到 GPU
            labels = labels.to(device)  # 将测试标签移动到 GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            error_count += (predicted!= labels).sum().item()
    accuracy = correct / total
    return accuracy, error_count


def main():
    global device  # 声明为全局变量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查 GPU 可用性
    trainloader, testloader = load_data()
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hidden_nodes_list = [500, 1000, 1500, 2000]
    results = []
    for lr in learning_rates:
        for hidden_nodes in hidden_nodes_list:
            model = CNN(hidden_nodes).to(device)  # 将模型移动到 GPU
            accuracy, error_count = train_and_evaluate(model, trainloader, testloader, lr)
            results.append((lr, hidden_nodes, accuracy, error_count))
            print(f"Learning Rate: {lr}, Hidden Nodes: {hidden_nodes}, Accuracy: {accuracy}, Error Count: {error_count}")



if __name__ == "__main__":
    main()