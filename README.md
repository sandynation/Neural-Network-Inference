# Neural-Network-Inference.cpp
#include <iostream>
#include <vector>
#include <cmath>

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<int> layers) {
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            weights.push_back(std::vector<std::vector<double>>(layers[i], std::vector<double>(layers[i + 1])));
            biases.push_back(std::vector<double>(layers[i + 1]));
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> activations = input;
        for (size_t i = 0; i < weights.size(); ++i) {
            activations = activate(matrix_vector_multiply(weights[i], activations) + biases[i]);
        }
        return activations;
    }

private:
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;

    std::vector<double> activate(const std::vector<double>& z) {
        std::vector<double> a(z.size());
        for (size_t i = 0; i < z.size(); ++i) {
            a[i] = 1.0 / (1.0 + std::exp(-z[i]));
        }
        return a;
    }

    std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    std::vector<double> matrix_vector_multiply(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector) {
        std::vector<double> result(matrix[0].size(), 0.0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[0].size(); ++j) {
                result[j] += matrix[i][j] * vector[i];
            }
        }
        return result;
    }
};

int main() {
    NeuralNetwork nn({3, 5, 2});

    std::vector<double> input = {1.0, 0.5, -1.2};
    std::vector<double> output = nn.forward(input);

    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# TensorFlow Model
class TensorFlowModel(tf.keras.Model):
    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# PyTorch Model
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(28*28, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.d1(x))
        return self.d2(x)

# Training TensorFlow Model
def train_tensorflow():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = TensorFlowModel()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)

# Training PyTorch Model
def train_pytorch():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = PyTorchModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

if __name__ == "__main__":
    print("Training with TensorFlow:")
    train_tensorflow()
    
    print("Training with PyTorch:")
    train_pytorch()
