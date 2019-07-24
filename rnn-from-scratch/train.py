from rnn import RNN
import numpy as np

corpus = open('input.txt', 'r').read()

chars = list(corpus)
unique_chars = set(chars)

print("Number of characters : ", len(chars))
print("Number of unique characters : ", len(unique_chars))

vocabulary_size = len(unique_chars)
sequence_length = 25

rnn = RNN(sequence_length, vocabulary_size)

char_to_idx = {ch: ind for ind, ch in enumerate(unique_chars)}
idx_to_char = {ind: ch for ind, ch in enumerate(unique_chars)}

data = {"chars": chars, "char_to_idx": char_to_idx, "idx_to_char": idx_to_char}

rnn.train(data, learning_rate=0.1)
