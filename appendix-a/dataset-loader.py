import torch
from torch.utils.data import Dataset, DataLoader

# Class representing a dataset, used by a dataloader
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

## Training data
train_features = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
train_labels = torch.tensor([0,0,0,1,1])
# Test data
test_features = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])
test_labels = torch.tensor([0,1])
train_dataset = ToyDataset(train_features, train_labels)
test_dataset = ToyDataset(test_features, test_labels)

print(len(train_dataset))

# Dataloader using the dataset
torch.manual_seed(123)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

for i, (features, labels) in enumerate(train_loader):
    print(f'Batch {i}:', features, labels)