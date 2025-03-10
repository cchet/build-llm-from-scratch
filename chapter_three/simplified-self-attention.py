import re
import torch
import tiktoken

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum()

# Calculate the weights as dot products for a query
inputs = torch.tensor([[0.43, 0.15, 0.89], # Your
                       [.55, .87, .66],    # journey
                       [.57, .85, .64],    # starts
                       [.22, .58, .33],    # with
                       [.77, .25, .1],     # one
                       [.05, .8, .55],     # step
                       ])
#  For the second input element only
query = inputs[1]
attention_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attention_scores_2[i] = torch.dot(x_i, query)
print(attention_scores_2)

# normalize values so they sum to one
attention_scores_2_tmp = attention_scores_2 / attention_scores_2.sum()
print('\nNormalized values:')
print(f'Attention weights: {attention_scores_2_tmp}')
print(f'Sum: {attention_scores_2_tmp.sum()}')
# normalize values with a naive implementation of softmax algorithm
attention_scores_2_tmp = softmax_naive(attention_scores_2)
print('\nNormalized values (naive softmax):')
print(f'Softmax: {attention_scores_2_tmp}')
print(f'Sum: {attention_scores_2_tmp.sum()}')
# normalize values with a torch implementation of softmax algorithm
attention_scores_2_tmp = torch.softmax(attention_scores_2, dim=0)
print('\nNormalized values (torch softmax):')
print(f'Softmax: {attention_scores_2_tmp}')
print(f'Sum: {attention_scores_2_tmp.sum()}')

# Calculate the context vector for the second input element
context_tensor = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_tensor += attention_scores_2_tmp[i] * x_i
print('\nContext tensor:')
print(context_tensor)

# Calculate for all input elements with for-loop
input_elements = inputs.shape[0]
attention_scores = torch.empty(input_elements, input_elements)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attention_scores[i, j] = torch.dot(x_i, x_j)
print('\nAttention scores:', attention_scores)
# Calculate for all input elements with matrix-multiplication
attention_scores = torch.matmul(inputs, inputs.T)
print('\nAttention scores (matrix-multiplication):', attention_scores)
# Normalize attention weights
normalized_attention_scores = torch.softmax(attention_scores, dim=-1)
print('\nNormalized attention scores:', normalized_attention_scores)
print('\nNormalized attention scores sum:', normalized_attention_scores.sum(dim=-1))
# Calculate the context tensor
context_tensor = torch.matmul(normalized_attention_scores, inputs)
print('\nContext tensor:', context_tensor)
