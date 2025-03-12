import re
import torch, torch.nn as nn

# Calculate the weights as dot products for a query
inputs = torch.tensor([[0.43, 0.15, 0.89], # Your
                       [.55, .87, .66],    # journey
                       [.57, .85, .64],    # starts
                       [.22, .58, .33],    # with
                       [.77, .25, .1],     # one
                       [.05, .8, .55],     # step
                       ])

x_2 = inputs[1] # The second word/input is the query
d_in = inputs.shape[1] # The inputs embedding size (=3)
d_out = 2 # the output embedding size (normally the same, but for following calculations set to '2')
## Initialize the weight matrices (require_grad would be tru if we would train the model, just for avoiding clutter)
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# Calculate the vectors
query_2 = torch.matmul(x_2, W_query)
key_2 = torch.matmul(x_2, W_key)
value_2 = torch.matmul(x_2, W_value)
print('Query_2: ', query_2)
print('Key_2: ', key_2)
print('Value_2: ', value_2)
# Calculate for all input
query = torch.matmul(inputs, W_query)
keys = torch.matmul(inputs, W_key)
values = torch.matmul(inputs, W_value)
print('Query: ', query)
print('Keys:', keys)
print('Values:', values)
# Calculate the attention score for W_22
keys_2 = keys[1]
att_score_22 = query_2.dot(keys_2)
print(att_score_22)
att_scores_2 = query_2 @ keys.T
print('Attention scores_2: ', att_scores_2)
# Calculate the attention weights
d_k = key_2.shape[-1]
attn_weights_2 = torch.softmax(att_scores_2 / d_k**0.5, dim=-1)
print('Embedding dimension: ', d_k)
print('Attention weights_2: ', attn_weights_2)
# Calculate the context vector_2
context_vector_2 = attn_weights_2 @ values
print('Context vectors_2: ', context_vector_2)

# Class implementing the slef-attention algorithm
class SelfAttention_V1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        query = x @ self.W_query
        values = x @ self.W_value
        att_scores = query @ keys.T
        attn_weights = torch.softmax(
            att_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attn_weights @ values
        return context_vector

# USing Linear instead of manually initializing the weight matrices
class SelfAttention_V2(nn.Module):
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

sa_V1 = SelfAttention_V1(d_in, d_out)
print('Context_vector_v1: ', sa_V1(inputs))
torch.manual_seed(789)
sa_V2 = SelfAttention_V2(d_in, d_out)
print('Context_vector_v2: ', sa_V2(inputs))
sa_V1.W_query = nn.Parameter(sa_V2.W_query.weight.T)
sa_V1.W_key = nn.Parameter(sa_V2.W_key.weight.T)
sa_V1.W_value = nn.Parameter(sa_V2.W_value.weight.T)
print('Context_vector_v1 (weights from V_2): ', sa_V1(inputs))
print('Context_vector_v2: ', sa_V2(inputs))