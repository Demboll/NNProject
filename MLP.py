import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#wybranie paczki do torcha z osbługą cuda - 12.3, przez pipa - NIE DA SIE
#ilosc warstw, do ilu rosnie dokladnosc, szerkokosc warstw x liczba parametrow, y dokladnosc, 2 wykresy do rozszerzania i zaglebiania

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper-parameters 
input_size = 32 * 32 * 3
hidden_size1 = 500
hidden_size2 = 200
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.001

# Dataset transformation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Data loading
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

examples = iter(test_loader)
example_data, example_targets = next(examples)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i].permute(1, 2, 0))
plt.show()

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1) 
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # should I use Adam?

# Lists to store loss and accuracy values
loss_list = []
accuracy_list = []

print_acc_interval = 100

# Training the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 3, 32, 32]
        # resized: [100, 3072]
        images = images.reshape(-1, 32 * 32 * 3).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Append loss and accuracy values to lists
        loss_list.append(loss.item())
        
        if (i + 1) % print_acc_interval == 0 or (i + 1) == n_total_steps:
            # Testing the model accuracy at the specified interval
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for test_images, test_labels in test_loader:
                    test_images = test_images.reshape(-1, 32 * 32 * 3).to(device)
                    test_labels = test_labels.to(device)
                    test_outputs = model(test_images)
                    _, predicted = torch.max(test_outputs.data, 1)
                    n_samples += test_labels.size(0)
                    n_correct += (predicted == test_labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                accuracy_list.append(acc)
        
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%')
        

# Testing the model accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 32 * 32 * 3).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

# Plotting the loss and accuracy graphs
plt.figure(figsize=(12, 4))

# Plotting the loss
plt.subplot(1, 2, 1)
plt.plot(loss_list, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()

# Plotting the accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy_list, label='Test Accuracy', color='orange')
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Iterations')
plt.legend()

plt.show()