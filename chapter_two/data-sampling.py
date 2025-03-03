import re
import torch
import tiktoken

def read_file(filename):
    # Read the input for the chapter_two
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self,text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


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
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired, ' | ', tokenizer.decode(context), "---->", tokenizer.decode([desired]))