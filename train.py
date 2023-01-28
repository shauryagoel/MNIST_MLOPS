import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml

# Load the config file
with open("config.yaml", "r") as f:
    config = yaml.load(f, yaml.Loader)

input_size = config["input_size"]
hidden_layer_size = config["hidden_layer_size"]
num_classes = config["num_classes"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]

# Get the MNIST dataset from torchvision
train_dataset = torchvision.datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="data", train=False, transform=transforms.ToTensor()
)

# Get the data loader from the above dataset
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# Get the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fully connected neural network with 2 weight layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


# Define the model
model = NeuralNet(input_size, hidden_layer_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start the training
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # resize from 100,1,28,28 to 100,784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{total_steps}], Loss: {loss.item():.4f}"
            )

# Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the model on the test set: {acc} %")
