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
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))


class AENN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = AENN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    for (img, __) in data_loader:
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch, img, recon))


for k in range(0, num_epochs, 4):
    plt.figure(figsize=(10, 3))  # slightly wider for text labels
    plt.gray()

    imgs = outputs[k][1].cpu().detach().numpy()
    recon = outputs[k][2].cpu().detach().numpy()

    # Top row: real images
    for i in range(9):
        plt.subplot(2, 9, i + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.text(-10, 14, 'Real', va='center', ha='right', fontsize=12, weight='bold')

    # Bottom row: predicted images
    for i in range(9):
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(recon[i][0], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.text(-10, 14, 'Predicted', va='center', ha='right', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.show()
