import os.path
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import collections
import matplotlib.pyplot as plt


def change_to_one_hot_label(labels):
    labels = np.array(labels)
    result = np.zeros((labels.size, 10))
    for idx, row in enumerate(result):
        row[labels[idx]] = 1
    return result


def get_data():
    if not os.path.exists('./data/train_set.npy'):
        train_set = datasets.MNIST('./data', train=True, download=True)
        test_set = datasets.MNIST('./data', train=False, download=True)
        train_labels = []
        for _, label in train_set:
            train_labels.append(label)
        test_labels = []
        for _, label in test_set:
            test_labels.append(label)
        train_set_array = train_set.data.numpy()
        test_set_array = test_set.data.numpy()

        train_set_array = train_set_array.reshape((train_set_array.shape[0], -1))
        test_set_array = test_set_array.reshape((test_set_array.shape[0], -1))
        train_labels = change_to_one_hot_label(train_labels)
        test_labels = change_to_one_hot_label(test_labels)

        with open('./data/train_set.npy', 'wb') as train_set:
            np.save(train_set, train_set_array)
        with open('./data/test_set.npy', 'wb') as test_set:
            np.save(test_set, test_set_array)

        with open('./data/train_labels.npy', 'wb') as train_labels_file:
            np.save(train_labels_file, train_labels)
        with open('./data/test_labels.npy', 'wb') as test_labels_file:
            np.save(test_labels_file, test_labels)

    elif os.path.exists('./data/train_set.npy'):
        with open('./data/train_set.npy', 'rb') as train_set_file:
            train_set_array = np.load(train_set_file)
        with open('./data/test_set.npy', 'rb') as test_set_file:
            test_set_array = np.load(test_set_file)

        with open('./data/train_labels.npy', 'rb') as train_labels_file:
            train_labels = np.load(train_labels_file)
        with open('./data/test_labels.npy', 'rb') as test_labels_file:
            test_labels = np.load(test_labels_file)
    train_set_array = train_set_array.reshape((train_set_array.shape[0], -1))
    test_set_array = test_set_array.reshape((test_set_array.shape[0], -1))
    return train_set_array, test_set_array, train_labels, test_labels


def softmax(softmax_input):
    softmax_input_max = np.max(softmax_input, axis=0)
    softmax_input = softmax_input - softmax_input_max
    softmax_input = np.exp(softmax_input)
    softmax_input_sum = np.sum(softmax_input, axis=0)
    result = softmax_input / softmax_input_sum
    return result


def cross_entropy(result, label):
    batch_size = result.shape[1]
    loss = -np.log(result[label, np.arange(batch_size)] + 1e-7)
    return loss


def sigmoid(sigmoid_input):
    sigmoid_input = 1 + np.exp(-sigmoid_input)
    result = 1 / sigmoid_input
    return result


class AffineNoBias:
    def __init__(self, weight):
        self.weight = weight
        self.x = None
        self.dw = None

    def forward(self, forward_input):
        self.x = forward_input
        return np.dot(self.weight, forward_input)

    def backward(self, dout):
        self.dw = np.dot(dout, self.x.T)
        dout = np.dot(self.weight.T, dout)
        return dout


class SigmoidClass:
    def __init__(self):
        self.forward_result = None

    def forward(self, forward_input):
        forward_input = sigmoid(forward_input)
        self.forward_result = forward_input
        return forward_input

    def backward(self, dout):
        dout = self.forward_result * (1 - self.forward_result) * dout
        return dout


class SoftMaxLoss:
    def __init__(self):
        self.result = None
        self.label = None

    def get_loss(self, forward_result, label):
        result = softmax(forward_result)
        label_arg = np.argmax(label, axis=0)
        loss = cross_entropy(result, label_arg)
        self.result = result
        self.label = label
        return loss

    def backward(self):
        dout = (self.result - self.label) / self.result.shape[0]
        return dout


class ThreeLayerNet:
    def __init__(self):
        self.loss = None
        self.dout = None
        self.lr = 5e-2
        self.params = {}
        self.params['w1'] = np.random.random((300, 784)) * 2 - 1
        self.params['w2'] = np.random.random((200, 300)) * 2 - 1
        self.params['w3'] = np.random.random((10, 200)) * 2 - 1

        # print(np.sum(self.params['w1']))
        # print(np.sum(self.params['w2']))
        # print(np.sum(self.params['w3']))
        #
        # print(self.params['w1'])
        # print(self.params['w2'])
        # print(self.params['w3'])

        self.layers = collections.OrderedDict()
        for i in range(1, 4):
            self.layers['affine' + str(i)] = AffineNoBias(self.params['w' + str(i)])
            self.layers['sigmoid' + str(i)] = SigmoidClass()
        self.last_layer = SoftMaxLoss()

    def predict(self, three_layer_input):
        for layer_name in self.layers:
            # print(layer_name)
            layer = self.layers[layer_name]
            three_layer_input = layer.forward(three_layer_input)
        return three_layer_input

    def get_loss(self, three_layer_input, label):
        three_layer_input = self.predict(three_layer_input)
        loss = self.last_layer.get_loss(three_layer_input, label)
        self.loss = loss
        return loss

    def accuracy(self, test_input, test_label):
        test_output = np.argmax(self.predict(test_input), axis=0)
        test_label_arg = np.argmax(test_label, axis=0)
        accuracy = np.sum(test_output == test_label_arg) / test_output.size
        return accuracy

    def errors(self, test_input, test_label):
        test_output = np.argmax(self.predict(test_input), axis=0)
        test_label_arg = np.argmax(test_label, axis=0)
        error_nums = np.sum(test_output != test_label_arg)
        return error_nums

    def get_gradient(self):
        layer_names = list(self.layers)
        layer_names.reverse()
        # print("softmax backward")
        dout = self.last_layer.backward()
        for layer_name in layer_names:
            # print("the backward name is ", layer_name)
            dout = self.layers[layer_name].backward(dout)  # print("the dout shape is ", dout.shape)

    def update_params(self):
        for num in range(1, 4):
            self.layers['affine' + str(num)].weight -= self.lr * self.layers['affine' + str(num)].dw


def main():
    batch_size = 100

    train_set_array, test_set_array, train_labels, test_labels = get_data()
    train_set_array = train_set_array.T
    test_set_array = test_set_array.T
    train_labels = train_labels.T
    test_labels = test_labels.T
    # print(train_set_array.shape)
    # print(test_set_array.shape)

    three_layer_net = ThreeLayerNet()
    tag = np.arange(0, 60000)
    lines = np.arange(0, 60000, batch_size)
    current_training_num = 0
    train_num = []
    test_accuracies = []
    test_count = 0
    for epoch in range(0, 10):
        np.random.shuffle(tag)
        for line in lines:
            test_count += 1
            current_training_num += batch_size
            three_layer_net.get_loss(train_set_array[:, tag[line:line + batch_size]], train_labels[:, tag[line:line + batch_size]])
            three_layer_net.get_gradient()
            three_layer_net.update_params()
            # if line == lines[-1]:
            if test_count % 100 == 0:
                print("the epoch is ", epoch)
                accuracy = 0
                for i in range(0, 10000, 100):
                    accuracy = accuracy + three_layer_net.accuracy(test_set_array[:, i:i + 100], test_labels[:, i:i + 100])
                print("the accuracy is ", accuracy)
                train_num.append(current_training_num)
                test_accuracies.append(accuracy)
    error_num = 0
    for i in range(0, 10000, 100):
        error_num = error_num + three_layer_net.errors(test_set_array[:, i:i + 100], test_labels[:, i:i + 100])
    print('The learning rate is ', three_layer_net.lr)
    print('The batch size is ', batch_size)
    print("The final number of test errors is ", error_num)
    print("The final test error rate is ", error_num / 10000)

    title = 'Learning curves when learning rate is %.0e and the batch size is %d.' % (three_layer_net.lr, batch_size)
    plt.style.use('seaborn')
    plt.plot(train_num, test_accuracies, label='Test error')
    plt.ylabel('Accuracy rate', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title, fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 100)
    plt.show()


if __name__ == '__main__':
    main()
