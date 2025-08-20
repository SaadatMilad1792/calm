# Compact Ai Language Model (CALM)
The Compact AI Language Model (CALM) is a lightweight transformer-based language model implemented in PyTorch. It is designed to demonstrate how modern autoregressive models work while staying simple enough to study and extend. CALM uses token embeddings, positional encodings, a transformer decoder, and a projection layer to perform next-token prediction and generate text sequences.

## Full Implementation

```python
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
## -- compact ai language model -- #####################################################################################
########################################################################################################################
class CompactAiLanguageModel(nn.Module):
  def __init__(self, parameters):
    super(CompactAiLanguageModel, self).__init__()
    # Store parameters
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

    # Move to device
    self = self.to(self.device)

    # Embedding layers: token and positional
    self.enc_embeddings = nn.Embedding(self.vocab_size, self.model_embd)
    self.enc_pos_encoding = nn.Embedding(self.block_size, self.model_embd)

    # Transformer decoder backbone
    self.decoder = TransformerDecoder(
      self.model_embd, self.num_heads, self.hidden,
      self.dropout_p, self.num_layers, self.device
    )

    # Output projection to vocabulary
    self.vocab_from_emb = nn.Linear(self.model_embd, self.vocab_size)

  # Batch generator: slices tokenized data into input-target pairs
  def get_batch(self, tokenized_data, batch_size, block_size):
    offset = torch.randint(len(tokenized_data) - block_size, (batch_size, ))
    X = torch.stack([tokenized_data[i:block_size + i] for i in offset])
    y = torch.stack([tokenized_data[(i + 1):(block_size + i + 1)] for i in offset])
    return X, y

  # Text generation: autoregressively samples new tokens
  def generate(self, X, max_new_tokens = 1):
    for token in range(max_new_tokens):
      X_slice = X[:, -self.block_size:]
      logits, loss = self(X_slice)
      logits = logits[:, -1, :]
      probas = F.softmax(logits, dim = -1)
      next_X = torch.multinomial(probas, num_samples = 1)
      X = torch.cat((X, next_X), dim = 1)
    return X
  
  # Training loop: performs forward, backward, optimization and validation
  def train(self, tokenized_data, optimizer, epochs = 1, verbose = True):
    tr_data, te_data, va_data = tokenized_data["tr"], tokenized_data["te"], tokenized_data["va"]
    for epoch in range(epochs):
      X, y = self.get_batch(tr_data, self.batch_size, self.block_size)
      logits, loss = self(X, y)
      optimizer.zero_grad(set_to_none = True)
      loss.backward()
      optimizer.step()

      # Save weights every epoch
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

  # Evaluation loop: averages loss across validation/test iterations
  @torch.no_grad()
  def eval(self, tokenized_data, iter = 100):
    total_loss = 0
    for _ in range(iter):
      X, y = self.get_batch(tokenized_data, self.batch_size, self.block_size)
      logits, loss = self(X, y)
      total_loss += loss
    return total_loss

  # Save model weights and parameters
  def save_weights(self, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    torch.save(self.state_dict(), path + "weights.pth")
    with open(path + "params.json", "w") as file:
      json.dump(self.params, file, indent = 2)
    return
```

## Explanation by Block

### 1. Imports
The code begins by importing **os** and **json** for file handling, **torch** and **torch.nn** for deep learning components, and a custom **TransformerDecoder** module which implements the core transformer building block.

### 2. Class Definition
The `CompactAiLanguageModel` class inherits from PyTorch's `nn.Module`. It encapsulates the entire language model, including embeddings, transformer decoder, and output layers.

### 3. Initialization
The `__init__` method takes a `parameters` dictionary and stores all hyperparameters (device, embedding size, number of heads, etc.).  
- It initializes **token embeddings** and **positional embeddings**.  
- It builds the **transformer decoder** backbone.  
- Finally, it defines a **linear projection layer** to map embeddings back to vocabulary logits.

### 4. Batch Generation
The `get_batch` method randomly slices the dataset into small training samples of length `block_size`. Each batch has input (`X`) and shifted target (`y`) sequences.

### 5. Text Generation
The `generate` method autoregressively generates new tokens. It repeatedly feeds the model’s own predictions back as input, extending the sequence one token at a time.

### 6. Training Loop
The `train` method performs the training process:  
- Fetch a batch, compute loss, backpropagate, and update weights.  
- Periodically evaluate on validation/test data.  
- Save weights at each epoch.

### 7. Evaluation
The `eval` method runs forward passes without gradients, averaging the loss over multiple iterations.

### 8. Saving
The `save_weights` method saves both the model weights (`weights.pth`) and hyperparameters (`params.json`) in human-readable format for reproducibility.

## Navigation Panel
Use this navigation panel to move forward or backward through the tutorial, or jump straight to the homepage whenever you like.<br>
[Proceed to the next section: Transformer Decoder](/documentation/markdowns/transformer_decoder.md)<br>
[Return to the previous section: Data Handler](/documentation/markdowns/data_handler.md)<br>
[Back to the table of contents](/)<br>