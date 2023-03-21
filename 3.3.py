import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def get_data():
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


class ThreeLayerNet(torch.nn.Module):
    def __init__(self):
        super(ThreeLayerNet, self).__init__()
        self.l1 = torch.nn.Linear(784, 300, bias=False)
        self.l2 = torch.nn.Linear(300, 200, bias=False)
        self.l3 = torch.nn.Linear(200, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return self.l3(x)




def main():
    train_loader, test_loader = get_data()
    model = ThreeLayerNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=9e-2)
    test_counts = 0
    current_train_num = 0
    train_num = []
    test_accuracies = []
    for epoch in range(10):
        for _, train_data in enumerate(train_loader, 0):
            test_counts+=1
            train_features, train_labels = train_data
            current_train_num += len(train_labels)
            optimizer.zero_grad()
            predictions = model(train_features)
            loss = criterion(predictions, train_labels)
            loss.backward()
            optimizer.step()
            if test_counts % 300==0:
                correct = 0
                total = 0
                with torch.no_grad():
                    for test_data in test_loader:
                        test_features, test_labels = test_data
                        test_pred = model(test_features)
                        _, test_predicted = torch.max(test_pred.data, dim=1)
                        total += test_labels.size(0)
                        correct += (test_predicted == test_labels).sum().item()
                accuracy = 100 * correct / total
                print('epoch %d, accuracy on test set: %d %% ' % (epoch, accuracy))
                train_num.append(current_train_num)
                test_accuracies.append(accuracy)

    correct = 0
    total = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_features, test_labels = test_data
            test_pred = model(test_features)
            _, test_predicted = torch.max(test_pred.data, dim=1)
            total += test_labels.size(0)
            correct += (test_predicted == test_labels).sum().item()
    accuracy = 100 * correct / total

    print("The final number of test errors is ", total - correct)
    print("The final test error rate is ", 100 * (total - correct) / total)


    plt.style.use('seaborn')
    plt.plot(train_num, test_accuracies, label='Test error')
    plt.ylabel('Accuracy rate', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 100)
    plt.show()




if __name__ == '__main__':
    main()
