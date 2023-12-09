import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ReadFromDataset import ReadFromDataset
from Model import ResNetModel, BottleneckBlock
from TrainingTesting import Training, Testing

random.seed(42)
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ReadFromDataset(root_dir='Dataset', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ResNetModel(BottleneckBlock, [2, 2, 2, 2])

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 5
print_frequency = 10
Training(model, train_loader, criterion, optimizer, num_epochs, print_frequency)


Testing(model, test_loader, print_frequency)
