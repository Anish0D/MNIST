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

# Rebuild the CNN model
class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = MCNN()
model.load_state_dict(torch.load('CNN_MNIST.pth'))  # Make sure filename matches!
model.eval()

# Load one test image and label
import random
random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]
image_tensor = image.unsqueeze(0)  # [1, 1, 28, 28]

# Predict
with torch.no_grad():
    output = model(image_tensor)
predicted_class = torch.argmax(output, dim=1).item()

# Show prediction
import matplotlib.pyplot as plt
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {predicted_class}, Actual: {label}")
plt.axis('off')
plt.show()