import torch

# Create tensor of different ranks
tensor0d = torch.tensor(2)
tensor1d = torch.tensor([1.0, 2.0, 3.0])
tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
tensor3d = torch.tensor([[[1, 2, 3]],
                         [[4, 5, 6]]])

# Print the tensors and their datatypes
print(tensor0d)
print(tensor0d.dtype)
print(tensor1d)
print(tensor1d.dtype)
print(tensor2d)
print(tensor2d.dtype)
print(tensor3d)
print(tensor3d.dtype)

# Convert tensor datatype
print('change dtype the tensor2d')
tensor2dvec = tensor2d.to(torch.float32)
print(tensor2dvec)
print(tensor2dvec.dtype)

# shape and reshape of the tensor
print('reshape the tensor2d')
print(tensor2d.shape)
print(tensor2d.reshape(3,2))
print(tensor2d.view(3,2))

# transpose a tensor
print('transpose the tensor2d')
print(tensor2d)
print(tensor2d.T)

# multiply tensor2d with its transposed tensor
print('multiplied the tensor2d with its transposed tensor')
print(tensor2d.matmul(tensor2d.T))
print(tensor2d @ tensor2d.T)