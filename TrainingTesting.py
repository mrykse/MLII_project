import torch


def Training(model, train_loader, criterion, optimizer, num_epochs, print_frequency):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch['image'], batch['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % print_frequency == print_frequency - 1:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {running_loss / print_frequency}")
                running_loss = 0.0

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


def Testing(model, test_loader, print_frequency):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, labels = batch['image'], batch['label']
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            if batch_idx % print_frequency == print_frequency - 1:
                print(f"Batch {batch_idx + 1}/{len(test_loader)}, Loss: {running_loss / print_frequency}")
                running_loss = 0.0

        print(f"Test Accuracy: {100 * correct / total:.2f}%")
        print(f"Average Test Loss: {running_loss / len(test_loader)}")


criterion = torch.nn.CrossEntropyLoss()
