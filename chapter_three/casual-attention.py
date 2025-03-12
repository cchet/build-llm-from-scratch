import re
import torch, torch.nn as nn
from torch.onnx.symbolic_opset11 import stack
from torch.onnx.symbolic_opset13 import diagonal

# Calculate the weights as dot products for a query
inputs = torch.tensor([[0.43, 0.15, 0.89], # Your
                       [.55, .87, .66],    # journey
                       [.57, .85, .64],    # starts
                       [.22, .58, .33],    # with
                       [.77, .25, .1],     # one
                       [.05, .8, .55],     # step
                       ])

d_in = inputs.shape[1] # The inputs embedding size (=3)
d_out = 2 # the output embedding size (normally the same, but for following calculations set to '2')

# USing Linear instead of manually initializing the weight matrices
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # The key weights for the whole input
        keys = self.W_key(x)
        # The query weights for the whole input
        query = self.W_query(x)
        # The values weights for the whole input
        values = self.W_value(x)
        # The attention scores for the query for all keys
        att_scores = query @ keys.T
        # The attention weights scaled by the square root
        attn_weights = torch.softmax(
            att_scores / keys.shape[-1]**0.5, dim=-1
        )
        # The context vector for the whole input
        context_vector = attn_weights @ values
        return context_vector

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # The key weights for the whole input
        keys = self.W_key(x)
        # The query weights for the whole input
        query = self.W_query(x)
        # The values weights for the whole input
        values = self.W_value(x)

        # The attention scores for the query for all keys
        att_scores = query @ keys.transpose(1, 2) # transpose dimension 1, 2 and keep batch dimension at position 0
        att_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # The attention weights scaled by the square root
        attn_weights = torch.softmax(
            att_scores / keys.shape[-1]**0.5, dim=-1
        )
        # Drop values from the weights
        attn_weights = self.dropout(attn_weights)
        # The context vector for the whole input
        context_vector = attn_weights @ values
        return context_vector

torch.manual_seed(789)
# Calculate the attention scores/weights unmasked
sa = SelfAttention(d_in, d_out)
queries = sa.W_query(inputs)
keys = sa.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print('Attention weights (unmasked): ', attn_weights)
context_length = attn_scores.shape[0]
# The mask we use to zero out the diagonal of the tensor
mask_simple = torch.tril(torch.ones(context_length, context_length))
print('Mask: ', mask_simple)
# Zero out the diagonal of the tensor
masked_simple = attn_weights * mask_simple
print('Attention weights (masked): ', masked_simple)
# normalize masked rows
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_normalized = masked_simple / row_sums
print('Row sums: ', row_sums)
print('Attention weights (masked, normalized): ', masked_simple_normalized)
####### Optimized masking in fewer steps
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print('Attention scores (masked): ', masked)
print('Attention weights (masked): ', attn_weights)
### Dropout masking
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # We want 50% dropout rate
example = torch.ones(6,6)
print('Dropout Example: ', dropout(example))
print('Dropout attn_weights: ', dropout(attn_weights))
### CasualAttention class usage
batch = torch.stack((inputs, inputs), dim=0)
print('Batch shape: ', batch.shape)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)
context_vectors = ca(batch)
print('context_vectors.shape: ', context_vectors.shape)

