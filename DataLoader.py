import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np

#przy ładowaniu danych ustawić formę danych - transforms np. do wektorów
#einops, do transformacji tensorów - rearrange, znamy co wchodzi do modelu mozemy to zmienic
#NEXT STEP - napisać modele
#CNN - rozbić na feature extractor i klasyfikator
#extractor wartswy konwolucyjne i pulling
#małe rozmiary zdjęcia - padding w kązdym zdjęciu, rozszerzamy zerami
#RNN - input size inny, prawdoodobnie 3 - kolory, hidden size - ile ma neuronow do zapisywania info, liczba warst, batch first - rozmiary sie zmieniaja
#biderectional na true fajnie

#LTSM


transform = transforms.Compose(
    [transforms.ToTensor(), ]) # Convert images to PyTorch tensors

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')

def show_images(images):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    images = images.numpy() * std[:, None, None] + mean[:, None, None]
    
    plt.figure(figsize=(8, 2))
    for i in range(images.shape[0]):
        plt.subplot(1, 4, i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.axis('off')
    plt.show()

for batch in train_loader:
    images, labels = batch
    show_images(images)
    break