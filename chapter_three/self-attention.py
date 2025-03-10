import re
import torch
import tiktoken

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
print(att_scores_2)