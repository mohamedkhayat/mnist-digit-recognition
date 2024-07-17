import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F


def train():
    loss = 0
    total_correct = 0
    total_samples = 0
    model.train()

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        correct = (torch.argmax(output, dim=1) == y).sum().item()
        total_correct += correct
        total_samples += y.size(0)
    accuracy = total_correct / total_samples
    loss = loss / len(train_loader)
    print('Train - Loss : {:.4f} Accuracy : {:.4f}'.format(loss, accuracy))


def validate():
    loss = 0
    total_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss += loss_function(output, y).item()
            correct = (torch.argmax(output, dim=1) == y).sum().item()
            total_correct += correct
            total_samples += y.size(0)
    accuracy = total_correct / total_samples
    loss = loss / len(valid_loader)
    print('Valid - Loss : {:.4f} Accuracy : {:.4f}'.format(loss, accuracy))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.set_float32_matmul_precision('high')
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

BATCH_SIZE = 32
N_CLASSES = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28

preprocess_trans = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT),
                                 scale=(.9, 1),
                                 ratio=(1, 1)),
    transforms.ColorJitter(brightness=.2, contrast=.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5, ), (0.5, ))
])

train_set.transform = preprocess_trans
valid_set.transform = preprocess_trans

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
train_N = len(train_loader.dataset)

valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
valid_N = len(valid_loader.dataset)
kernel_size = 3

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=1),
    nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0), nn.MaxPool2d(2, stride=2),
    nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1),
    nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2, stride=2),
    nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1),
    nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0), nn.MaxPool2d(2, stride=2),
    nn.Flatten(), nn.Linear(128 * 3 * 3, 512), nn.ReLU(), nn.Linear(512, 512),
    nn.ReLU(), nn.Linear(512, N_CLASSES))

model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

epochs = 10

for epoch in range(epochs):
    print('Epoch : {}'.format(epoch))
    train()
    validate()

user_input = int(
    input("Press 1 if you want to save the model, any other number to exit: "))
if user_input == 1:
    torch.save(model, 'model.pth')
    print("Model saved.")
else:
    print("Model not saved.")
