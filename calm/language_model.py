########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer_decoder import TransformerDecoder

########################################################################################################################
## -- bigram language model -- #########################################################################################
########################################################################################################################
class CompactAiLanguageModel(nn.Module):
  def __init__(self, parameters):
    super(CompactAiLanguageModel, self).__init__()
    self.params = parameters
    self.device = parameters["device"]
    self.val_iter = parameters["val_iter"]
    self.val_freq = parameters["val_freq"]
    self.save_path = parameters["save_path"]
    self.num_heads = parameters["num_heads"]
    self.model_embd = parameters["model_embd"]
    self.vocab_size = parameters["vocab_size"]
    self.batch_size = parameters["batch_size"]
    self.block_size = parameters["block_size"]
    self.hidden = parameters["hidden"]
    self.dropout_p = parameters["dropout_p"]
    self.num_layers = parameters["num_layers"]

    self = self.to(self.device)
    self.enc_embeddings = nn.Embedding(self.vocab_size, self.model_embd)
    self.enc_pos_encoding = nn.Embedding(self.block_size, self.model_embd)
    self.decoder = TransformerDecoder(self.model_embd, self.num_heads, self.hidden, 
                                      self.dropout_p, self.num_layers, self.device)

    self.vocab_from_emb = nn.Linear(self.model_embd, self.vocab_size)

  def get_batch(self, tokenized_data, batch_size, block_size):
    offset = torch.randint(len(tokenized_data) - block_size, (batch_size, ))
    X = torch.stack([tokenized_data[i:block_size + i] for i in offset])
    y = torch.stack([tokenized_data[(i + 1):(block_size + i + 1)] for i in offset])
    return X, y

  def generate(self, X, max_new_tokens = 1):
    for token in range(max_new_tokens):
      X_slice = X[:, -self.block_size:]
      logits, loss = self(X_slice)
      logits = logits[:, -1, :]
      probas = F.softmax(logits, dim = -1)
      next_X = torch.multinomial(probas, num_samples = 1)
      X = torch.cat((X, next_X), dim = 1)
    return X
  
  def train(self, tokenized_data, optimizer, epochs = 1, verbose = True):
    tr_data, te_data, va_data = tokenized_data["tr"], tokenized_data["te"], tokenized_data["va"]
    for epoch in range(epochs):
      X, y = self.get_batch(tr_data, self.batch_size, self.block_size)
      logits, loss = self(X, y)
      optimizer.zero_grad(set_to_none = True)
      loss.backward()
      optimizer.step()

      self.save_weights(path = self.save_path)
      if epoch % self.val_freq == 0 and verbose:
        print(f"tr_loss: {loss.item():.4f}", end = "")
        if va_data is not None:
          va_loss = self.eval(va_data, self.val_iter)
          print(f", va_loss: {va_loss.item():.4f}", end = "")
        if te_data is not None:
          te_loss = self.eval(te_data, self.val_iter)
          print(f", te_loss: {te_loss.item():.4f}", end = "")
        print()

  @torch.no_grad()
  def eval(self, tokenized_data, iter = 100):
    total_loss = 0
    for _ in range(iter):
      X, y = self.get_batch(tokenized_data, self.batch_size, self.block_size)
      logits, loss = self(X, y)
      total_loss += loss
    return total_loss

  def save_weights(self, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    torch.save(self.state_dict(), path + "weights.pth")
    with open(path + "params.json", "w") as file:
      json.dump(self.params, file, indent = 2)
    return

  def load_weights(self, path, device = 'cpu'):
    state_dict = torch.load(path, map_location = torch.device(device))
    self.load_state_dict(state_dict)
    return

  def forward(self, X, y = None):
    B, T = X.shape
    embeddings = self.enc_embeddings(X)
    positional_encoding = self.enc_pos_encoding(torch.arange(T).to(self.device))
    x = embeddings + positional_encoding
    x = self.decoder(x, mask = "auto")
    logits = self.vocab_from_emb(x)
    if y is not None:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      y = y.view(B * T)
      loss = F.cross_entropy(logits, y)
    else:
      loss = None
    
    return logits, loss
