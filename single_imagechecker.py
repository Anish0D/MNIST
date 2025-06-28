import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import random

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

# USING AN OPTIMIZER (GRADIENT DESCENT)
model = Classifier() #Creates one real instance of my machine

model.load_state_dict(torch.load('mnist_model.pth'))

random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]

# Pick one image and label from the test dataset
image, label = test_dataset[random_index]  # 0 can be any index from 0 to len(test_dataset)-1

# Transform the image (already normalized in test_dataset, so just turn into a batch)
image_tensor = image.unsqueeze(0)  # shape becomes [1, 1, 28, 28]

# Put model in evaluation mode
model.eval()

# Pass the image to the model
with torch.no_grad():
    output = model(image_tensor)  # shape: [1, 10]

# Get predicted class (index of highest score)
predicted_class = torch.argmax(output, dim=1).item()

print(f"Predicted: {predicted_class}, Actual: {label}")

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_class}, Actual: {label}")
plt.axis('off')
plt.show()