# APPENDIX A / page: 262
import torch
import torch.nn.functional as F
from torch.autograd import grad

# single layer network for logistic regression classification
y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1])  # input feature
w1 = torch.tensor([2.2], requires_grad=True)  # weight parameter
b = torch.tensor([0.0], requires_grad=True)  # bias input

z = x1 * w1 + b  # net input
a = torch.sigmoid(z)  # activation and output

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print('\nDirectly calculating the gradients')
print(f'Gradient for w1: {grad_L_w1}')
print(f'Gradient for b:  {grad_L_b}')

print('\nCalculating the gradients via loss.backward()')
loss.backward()
print(f'Gradient for w1: {w1.grad}')
print(f'Gradient for b:  {b.grad}')
