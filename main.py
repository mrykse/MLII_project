import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from ReadFromDataset import ReadFromDataset
from Model import ResNetModel, BasicBlock

torch.manual_seed(42)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ReadFromDataset(root_dir='Dataset', transform=transform)

# 0.8, 0.1, 0.1 for train, val, test
train_size = int(0.8 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=136, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=136, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=136, shuffle=False)

# ResNet26 & BasicBlock
model = ResNetModel(BasicBlock, [2, 2, 2, 2], num_classes=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
num_epochs = 2
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    batch_counter = 0

    for batch_idx, batch in enumerate(train_loader):
        inputs, labels = batch['image'], batch['label']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        train_accuracy = total_correct / total_samples

        # Print batch-level logs for every 15 batches
        if batch_counter % 15 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Training Accuracy: {train_accuracy}, Loss: {loss.item()}",
                flush=True)

            # Save training accuracy for plotting
            train_accuracies.append(train_accuracy)

        batch_counter += 1

    # Save training accuracy for plotting (for the last batch in the epoch)
    train_accuracies.append(train_accuracy)

    # Testing on the validation set
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch in val_loader:
            inputs, labels = batch['image'], batch['label']
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        val_accuracy = total_correct / total_samples

    # Print epoch-level logs
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")

    # Save validation accuracy for plotting
    val_accuracies.append(val_accuracy)

# Plotting the training accuracies after the training loop
x_axis_data = range(1, len(train_accuracies) * 15 + 1, 15)
plt.plot(x_axis_data, train_accuracies, label='Training Accuracy')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.savefig('training_accuracy_plot.png')
plt.show()

# Testing on the test set
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for batch in test_loader:
        inputs, labels = batch['image'], batch['label']
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    test_accuracy = total_correct / total_samples
    print(f"Test Accuracy: {test_accuracy}")
