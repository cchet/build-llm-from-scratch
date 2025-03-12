import torch, torch.nn as nn

##############################################################################
## This is the main python file where we implement the LLM
##############################################################################

## The dataset class for loading the data and creating batches of it
class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # The vocabulary based on training text
        token_ids = tokenizer.encode(text)

        # Loop over text, with a sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            # The input for training
            input_chunk = token_ids[i:i + max_length]
            # The expected predicted tokens for the training token
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # Transform the input and target_ids to tensors
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

## The casual attention class for the self-attention
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