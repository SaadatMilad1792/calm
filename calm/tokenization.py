########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import pickle
from tqdm import tqdm

########################################################################################################################
## -- tokenizer training, encoder, and decoder -- ######################################################################
########################################################################################################################
class Tokenizer():
  def __init__(self, text_data = None):
    super(Tokenizer, self).__init__()
    self.new_merged_indexes = None
    self.text_data = text_data
    self.string_encoding = "utf-8"
    self.save_state_path = None
    self.base_vocab_size = 256
    self.max_vocab_size = 256
    self.vocab = None

  def get_base_tokens(self, text_data):
    text_data = text_data.encode(self.string_encoding)
    return list(map(int, text_data))

  def get_counts(self, indexes):
    counter_obj = {}
    for pair in zip(indexes[:-1], indexes[1:]):
      counter_obj[pair] = counter_obj.get(pair, 0) + 1
    return counter_obj
  
  def get_most_common_pair(self, counter_obj):
    return max(counter_obj, key = counter_obj.get)
  
  def merge_to_new_index(self, indexes, pattern, new_index):
    new_index_list, skip_flag = [], False
    for i in range(len(indexes) - 1):
      if skip_flag == False and indexes[i] == pattern[0] and indexes[i + 1] == pattern[1]:
        new_index_list.append(new_index)
        skip_flag = True
      else:
        if skip_flag == False:
          new_index_list.append(indexes[i])
        skip_flag = False
    new_index_list.append(indexes[-1]) if skip_flag == False else None
    return new_index_list
  
  def generate_vocab(self, save_state = True):
    base_tokens = self.get_base_tokens(self.text_data)
    base_counts = self.get_counts(base_tokens)
    self.vocab = {index:bytes([index]) for index in range(self.base_vocab_size)}
    self.new_merged_indexes = {}

    for i in tqdm(range(self.base_vocab_size, self.max_vocab_size), desc="Merging BPE pairs"):
      most_common_pair = self.get_most_common_pair(base_counts)
      base_tokens = self.merge_to_new_index(base_tokens, most_common_pair, i)
      self.new_merged_indexes[most_common_pair] = i
      base_counts = self.get_counts(base_tokens)

    for (t0, t1), index in self.new_merged_indexes.items():
      self.vocab[index] = self.vocab[t0] + self.vocab[t1]

    if save_state:
      self.save_state(path = self.save_state_path)

    return
  
  def save_state(self, path = None):
    if path is None:
      path = "../data/vocab/tokenizer_state.pkl"
      os.makedirs(os.path.dirname(path), exist_ok = True)
      
    state_object = {
      "string_encoding": self.string_encoding,
      "base_vocab_size": self.base_vocab_size,
      "max_vocab_size": self.max_vocab_size,
      "new_merged_indexes": self.new_merged_indexes,
      "vocab": self.vocab
    }

    with open(path, "wb") as file:
      pickle.dump(state_object, file)

    return

  def load_state(self, path):
    with open(path, "rb") as file:
      state_object = pickle.load(file)

    self.string_encoding = state_object["string_encoding"]
    self.base_vocab_size = state_object["base_vocab_size"]
    self.max_vocab_size = state_object["max_vocab_size"]
    self.new_merged_indexes = state_object["new_merged_indexes"]
    self.vocab = state_object["vocab"]
    return

  def encoder(self, text):
    tokens = list(text.encode(self.string_encoding))
    while len(tokens) >= 2:
      counts = self.get_counts(tokens)
      replacement_pair = min(counts, key = lambda x: self.new_merged_indexes.get(x, float("inf")))
      if replacement_pair not in self.new_merged_indexes:
        break
      tokens = self.merge_to_new_index(tokens, replacement_pair, self.new_merged_indexes[replacement_pair])
    return tokens

  def decoder(self, tokens):
    return b''.join(self.vocab[token] for token in tokens).decode(self.string_encoding, errors = "replace")
