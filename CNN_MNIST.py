import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

## Using my CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## LOADING DATA

# Making the data digestible for the nn to use by tranforming and normalizing it
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # these are the stdevs and means of the dataset so it can be normalized
])

# Download training data
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Load it in batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

## Creating the CNN architecture

class MCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MCNN, self).__init__()
        ## 2 Convolution layers to pick up on the edge and specific details of the images
        ## There is a pool function, but 2 pooling layers in the real network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## 2 Sets of Neural Networks nodes, one section to go from the large amount of input nodes from the conv
        ## the second section goes to the 10 output values
        ## from there the probabilities are calculated to match it to the possible label
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## Begins an instance of the Neural network
model = MCNN().to(device)

## Creates the loss and optimizer for the 
loss_calc = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

## I wanna measure accuracy and do a similar graph as the Classifier so Im copying the code
# An Evaluation Function for the Model on testing data
def evaluate(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

## Turn on Training Mode >:))
model.train()

# Creating a training loop to run the program through multiple times (epochs)
accuracies = []
for epoch in range(0, 31):
     # keeping track of loss
    train_loss = 0
    test_loss = 0

    # I want to track accuracy as well
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
     
        outputs = model(images)
        loss = loss_calc(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

         # I ChatGPTed this for an accuracy calculation
        _, predicted = torch.max(outputs, 1)         # get the index of the highest score for each image
        correct += (predicted == labels).sum().item()  # count how many were right
        total += labels.size(0)                       # total images in this batch

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_calc(outputs, labels)

        test_loss += loss.item()

         # I ChatGPTed this for an accuracy calculation
        _, predicted = torch.max(outputs, 1)         # get the index of the highest score for each image
        correct += (predicted == labels).sum().item()  # count how many were right
        total += labels.size(0)                       # total images in this batch

    print(f"Epoch {epoch+1}, Loss: {test_loss:.4f}")

    # Keeps track of the loss per epoch AND the accuracy too, so we can see how good it is
    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f"Epoch {epoch+1}, Loss:{test_loss:.4f}, Accuracy:{accuracy:.2f}%") # the .4 and .2 are just how many numbers i want past the decimal points 

# Save the trained model
torch.save(model.state_dict(), 'CNN_MNIST.pth')
print("Model saved as CNN_MNIST.pth")


# Graphing accuracy growth
plt.plot(range(1, 31), accuracies)
plt.ylabel('Accuracy in %')
plt.xlabel('Number of epochs run')
plt.show()
    