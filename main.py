import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


## LOADING DATA

#Making the data digestible for the nn to use by tranforming and normalizing it
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

classes = ('zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven',
           'eight', 'nine')


#CREATING ARCHITECTURE

# Building the model from the nn.Module base that pytorch has
class Classifier(nn.Module):
    #Initializing the neural network, this one has 2 hidden layers and 1 input/output layer
    def __init__(self):
        super().__init__()
        # All these next 3 lines are doing is setting up how the fully connected system looks layer by layer
        # First layer takes the 784 1D Vector and connects it to a 128 node layer
        self.fc1=nn.Linear(28*28, 128)
        # Second layer takes the 128 node layer and connects it to a 64 node layer
        self.fc2=nn.Linear(128, 64)
        # Third layer brings the 64 node layer to the output layer
        self.fc3=nn.Linear(64, 10)

    # This is the function that moves the input through the system
    def forward(self, x):
        # Reshapes the batch into a 1D vector, so it becomes digestible
        x = x.view(x.size(0), -1)
        # These next few lines just add a weight function to the inputs
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        # This line is the output, with raw scores
        x = self.fc3(x)
        #Automatic softmax function
        return x


# DEFINE A LOSS FUNCTION 
loss_fn = nn.CrossEntropyLoss()

# USING AN OPTIMIZER (GRADIENT DESCENT)
model = Classifier() #Creates one real instance of my machine
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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

# TRAINING LOOP TIME >:)

model.train()

# The number of times im going to test it (in this case 20)
accuracies = []
for epoch in range(50):
    # keeping track of loss
    total_loss = 0

    # I want to track accuracy as well
    correct = 0
    total = 0
    # Loading the mini-batches (64 images) and match them to their labels
    for images, labels in train_loader:

        # Runs the input through my Classifier function and gives output
        outputs = model(images)

        # measuring the loss with the crossentropyloss function
        loss = loss_fn(outputs, labels)

        # optimizing with gradient descent
        optimizer.zero_grad()

        # Backpropogation to check which weights are effecting the outcome and how
        loss.backward()

        # Optimizes the data by tweaking the weights to get better
        optimizer.step()

        #Tracks how good the model is going throughout the run
        total_loss += loss.item()

        # I ChatGPTed this for an accuracy calculation
        _, predicted = torch.max(outputs, 1)         # get the index of the highest score for each image
        correct += (predicted == labels).sum().item()  # count how many were right
        total += labels.size(0)                       # total images in this batch

    # Keeps track of the loss per epoch AND the accuracy too, so we can see how good it is
    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f"Epoch {epoch+1}, Loss:{total_loss:.4f}, Accuracy:{accuracy:.2f}%") # the .4 and .2 are just how many numbers i want past the decimal points 

# After training finishes, evaluate on test data
test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved as mnist_model.pth")


# Graphing accuracy growth
plt.plot(range(1, 51), accuracies)
plt.ylabel('Accuracy in %')
plt.xlabel('Number of epochs run')
plt.show()