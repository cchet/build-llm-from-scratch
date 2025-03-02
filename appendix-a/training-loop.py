import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

### The neural network implementation
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(), # non-linear-activation functions between layers

            # 2nd hidden layer
            torch.nn.Linear(30, 20), # input nodes must match output nodes of former layer
            torch.nn.ReLU(), # non-linear-activation functions between layers

            # output layer
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        return self.layers(x) # output of last layer are called logits

### The dataset implementation
class ToyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        one_feature = self.features[index]
        one_label = self.labels[index]
        return one_feature, one_label

    def __len__(self):
        return self.labels.shape[0]

### Functions which calculates model accuracy
def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

### Training data
train_features = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
train_labels = torch.tensor([0,0,0,1,1])
### Test data
test_features = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])
test_labels = torch.tensor([0,1])

torch.manual_seed(123)
train_dataset = ToyDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_dataset = ToyDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3
# Training loop to train the model
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Logging
        print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}'
              f' | Batch: {batch_idx+1:03d}/{len(train_loader):03d}'
              f' | Loss: {loss:.2f}')

# Evaluate the nodel
model.eval()
with torch.no_grad():
    outputs = model(train_features)

torch.set_printoptions(sci_mode=False)
probabilities = torch.softmax(outputs, dim=1)
print(probabilities)
probability_tensor = torch.argmax(outputs, dim=1)
print(probability_tensor)

print(compute_accuracy(model, train_loader))
print(compute_accuracy(model, test_loader))

torch.save(model.state_dict(), 'neuralnet-weights.pth')
