import torch

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

model = NeuralNetwork(50, 3)
print(f'NeuralNetwork structure \n{model}')

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {num_params}')
print(f'Model layer[0].wight.shape: {model.layers[0].weight.shape}')
print(f'Model layer[0].bias.shape: {model.layers[0].bias.shape}')

# This is how we would traint he model with backward propagation
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(f'\n{model.layers[0].weight}')
x = torch.randn(1, 50)
out = model(x)
print(f'Output: {out}')

# For inference, we deactivate the backward-propagation
with torch.no_grad():
    out = model(x)
    print(f'Nograd-Output: {out}')
    out = torch.softmax(out, dim=1) # so the putput is represented as probabilities of 0-1 with the sum=1
    print(f'Softmax-Output: {out}')
