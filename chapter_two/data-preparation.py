import re
import torch
import tiktoken


def read_file(filename):
    # Read the input for the chapter_two
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


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


def create_dataloader(text, batch_size=4, max_length=256,
                      stride=128, shuffle=True, drop_last=True,
                      num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return dataloader

### Experiments with tiktokenizer
tokenizer = tiktoken.get_encoding('gpt2')
text = read_file('../the-verdict.txt')

# Next word prediction training
encoded_text = tokenizer.encode(text)
print(len(encoded_text))
enc_sample = encoded_text[50:]
print(len(enc_sample))
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print(f'x: {x}')
print(f'y:      {y}')
# The next word which is expected to be predicted by the model
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired, ' | ', tokenizer.decode(context), "---->", tokenizer.decode([desired]))
print('')

### Using GPTDataset and loader
dataloader = create_dataloader(text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iterator = iter(dataloader)
inputs, targets = next(data_iterator)
print(f'Inputs: {inputs}')
print(f'Target: {targets}')

# Embedding small inputs with small dimensions
torch.manual_seed(123)
inputs = torch.tensor([2,3,5,1])
embedding_layer = torch.nn.Embedding(num_embeddings=6, embedding_dim=3)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(inputs))

# Embedding large inputs with 256 dimensions
max_length = 4
output_dimension = 256
embedding_layer = torch.nn.Embedding(num_embeddings=tokenizer.n_vocab, embedding_dim=output_dimension)
dataloader = create_dataloader(text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iterator = iter(dataloader)
inputs, targets = next(data_iterator)
token_embeddings = embedding_layer(inputs)
print(f'Inputs\n: {inputs}')
print(f'Inputs.Shape\n: {inputs.shape}')
context_length = max_length
pos_embedding_layer =  torch.nn.Embedding(num_embeddings=context_length, embedding_dim=output_dimension)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


