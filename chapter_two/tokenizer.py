import re
import tiktoken

class Tokenizer:
    token_unknown = '<|unk|>'
    token_endoftext = '<|endoftext|>'

    def __init__(self, vocab):
        # The vocabulary mapping words to integers (maps tokens to integer for tensors)
        self.str_to_int = vocab
        # The inverse mapping integer to words
        self.int_to_str = {id:word for word,id in vocab.items()}

    def encode(self, text):
        # Split the input text to words, special characters, whitespaces
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        # Remove all whitespace items
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace unkown words with special token
        preprocessed = [item if item in self.str_to_int else self.token_unknown for item in preprocessed]
        # Map the words to integer representations in the vocabulary
        ids = [self.str_to_int[word] for word in preprocessed]
        return ids

    def decode(self, ids):
        # Build the text by mapping integers back to words and special characters
        text = " ".join([self.int_to_str[id] for id in ids])
        # Remove spaces before defined punctuations
        text = re.sub(r'\s+([.,?!"()\'])', r'\1', text)
        return text

def read_file(filename):
    # Read the input for the chapter_two
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def create_vocabulary(filename):
    raw_text = read_file(filename)
    # Split the input text
    split_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    # Cleanup text of whitespaces
    split_text_no_whitespace = [item.strip() for item in split_text if item.strip()]
    # Make a set and sort alphanumerically
    sorted_split_text_no_whitespace = sorted(set(split_text_no_whitespace))
    sorted_split_text_no_whitespace.extend([Tokenizer.token_endoftext, Tokenizer.token_unknown])
    # Create the vocabulary mapping the words to integers
    return {word:integer for integer,word in enumerate(sorted_split_text_no_whitespace)}

# Vocabulary for tokenization
vocabulary = create_vocabulary('../the-verdict.txt')
print(len(vocabulary))
print(vocabulary)
print('')

# The tokenizer instance
tokenizer = Tokenizer(vocabulary)

# Encode/Decode known texts
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
known_text_ids = tokenizer.encode(text)
print(text)
print(known_text_ids)
print(tokenizer.decode(known_text_ids))
print('')

# Encode/Decode unkown text
text_one = 'Hello, do you like tea?'
text_two = 'In the sunlit terraces of the palace.'
text = Tokenizer.token_endoftext.join([text_one, text_two])
unknown_text_ids = tokenizer.encode(text)
print(text)
print(unknown_text_ids)
print(tokenizer.decode(unknown_text_ids))
print('')

# Tiktoken examples
tokenizer_toktoken = tiktoken.get_encoding('gpt2')
tiktoken_ids = tokenizer_toktoken.encode(text, allowed_special={Tokenizer.token_endoftext})
print(text)
print(tiktoken_ids)
print(tokenizer_toktoken.decode(tiktoken_ids))
print('')

# Example of unkown words for tiktoken
unknown_word_text = 'Akwirw ier'
unknown_text_ids = tokenizer_toktoken.encode(unknown_word_text)
print(unknown_word_text)
print(unknown_text_ids)
for token in unknown_text_ids:
    print(token)
    print(tokenizer_toktoken.decode([token]))
print(tokenizer_toktoken.decode(unknown_text_ids))
print('')