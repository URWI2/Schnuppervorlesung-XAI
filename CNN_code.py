
"""
Es wird für diese Aufgabe eine PythonUmgebung benötigt. 
Sie können die benötigete Umgebung in miniconda mit den folgenden Befehlen aufsetzen: 
    
conda create --name xai
conda activate xai
pip install torch torchvision captum matplotlib==3.3.4 Flask-Compress numpy scikit-learn pandas
pip install spyder-kernels==2.4.*

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# Load data from sklearn
mnist = fetch_openml('mnist_784')

data = mnist.data.to_numpy().reshape(-1, 1, 28, 28).astype(np.float32) / 255  # normalize the data to [0, 1]
targets = mnist.target.to_numpy().astype(np.int64)  # targets are the labels

# Split data into training and testing sets
train_data, test_data = data[:60000], data[60000:]
train_targets, test_targets = targets[:60000], targets[60000:]

train_dataset = MNISTDataset(train_data, train_targets)
test_dataset = MNISTDataset(test_data, test_targets)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# a simple CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

model = CNN()


# Training loop
def train(model, train_loader, epochs=2):
    
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

# Testing loop
def test(model, test_loader):
    
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Train the model
train(model, train_loader, epochs=2)

# Test the model
test(model, test_loader)

model.eval()



"""
Test one Instance 
"""

data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
data_iter = iter(data_loader)


test_img, label = next(data_iter)

test_img_data = np.asarray(test_img)
test_img_data = np.squeeze(test_img_data )
plt.imshow(test_img_data)
plt.show()

output = model(test_img)

pred = output.argmax()



"""
Saliency Map
"""

