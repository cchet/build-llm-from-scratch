import re

def read_and_tokenize_input(filename):
    # Read the input for the llm
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    # Split the input text
    split_text = re.split(r'([.,:;?!_"()\']|--|\s)', raw_text)
    split_text_no_whitespace = [item for item in split_text if item.strip()]
    print(f'Raw input:{raw_text[:30]}')
    print(f'Split input: {split_text}')
    print(f'Normalized split input: {split_text_no_whitespace}')

read_and_tokenize_input('the-verdict.txt')