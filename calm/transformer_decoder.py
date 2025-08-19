########################################################################################################################
## -- libraries and packages -- ########################################################################################
########################################################################################################################
import math
import torch
import torch.nn as nn

########################################################################################################################
## -- multi-headed attention mechanism -- ##############################################################################
########################################################################################################################
class MultiHeadedAttention(nn.Module):
  def __init__(self, model_emb, num_heads, dropout_p = 0.0):
    super(MultiHeadedAttention, self).__init__()
    self.num_heads = num_heads
    self.model_emb = model_emb
    self.dropout_1 = nn.Dropout(p = dropout_p)
    self.dropout_2 = nn.Dropout(p = dropout_p)
    self.heads_emb = model_emb // num_heads
    self.qkv_extractor = nn.Linear(self.model_emb, 3 * self.model_emb)
    self.qkv_connector = nn.Linear(self.model_emb, 1 * self.model_emb)

  def scaled_attention(self, q, k, v, mask = None):
    att_mat = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    att_mat = self.dropout_1(att_mat)
    att_mat = nn.functional.softmax(att_mat + mask.unsqueeze(1) if mask is not None else att_mat, dim = -1)
    return torch.matmul(att_mat, v), att_mat

  def causal_mask(self, batch_size, block_size):
    mask = torch.triu(torch.ones(batch_size, block_size, block_size), diagonal = 1).bool()
    mask = torch.where(mask, float("-inf"), 0)
    return mask

  def forward(self, x, mask = "auto"):
    batch_size, block_size, model_emb = x.shape
    mask = (self.causal_mask(batch_size, block_size) if mask == "auto" else mask)
    qkv_vector = self.qkv_extractor(x)
    qkv_vector = qkv_vector.reshape(batch_size, block_size, self.num_heads, 3 * self.heads_emb)
    qkv_vector = qkv_vector.permute(0, 2, 1, 3)
    q, k, v = qkv_vector.chunk(3, dim = -1)
    values, att_mat = self.scaled_attention(q, k, v, mask)
    values = values.permute(0, 2, 1, 3).reshape(batch_size, block_size, self.model_emb)
    values = self.qkv_connector(values)
    values = self.dropout_2(values)
    return values, att_mat
  
class LayerNorm(nn.Module):
  def __init__(self, dim, eps = 1e-5):
    super(LayerNorm, self).__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True)
    var = x.var(dim = -1, unbiased = False, keepdim = True)
    x = (x - mean) / (torch.sqrt(var + self.eps))
    return self.gamma * x + self.beta

class FeedForward(nn.Module):
  def __init__(self, model_emb, hidden, dropout_p):
    super(FeedForward, self).__init__()
    self.ffwd_1 = nn.Linear(model_emb, hidden)
    self.ffwd_2 = nn.Linear(hidden, model_emb)
    self.dropout = nn.Dropout(p = dropout_p)
    self.ReLU = nn.ReLU()

  def forward(self, x):
    x = self.ffwd_1(x)
    x = self.ReLU(x)
    x = self.dropout(x)
    x = self.ffwd_2(x)
    return x
  
class DecoderSequential(nn.Sequential):
  def forward(self, *inp):
    x, mask = inp
    for module in self._modules.values():
      x = module(x, mask)
    return x

class DecoderBlock(nn.Module):
  def __init__(self, model_emb, num_heads, hidden, dropout_p):
    super(DecoderBlock, self).__init__()
    self.multi_head_att = MultiHeadedAttention(model_emb, num_heads, dropout_p)
    self.feed_forward = FeedForward(model_emb, hidden, dropout_p)
    self.dropout_1 = nn.Dropout(p = dropout_p)
    self.dropout_2 = nn.Dropout(p = dropout_p)
    self.layer_norm_1 = LayerNorm(model_emb)
    self.layer_norm_2 = LayerNorm(model_emb)

  def forward(self, x, mask = None):
    x_skip = x.clone()
    x = self.layer_norm_1(x)
    x, _ = self.multi_head_att(x, mask = mask)
    x = x + x_skip
    x = self.dropout_1(x)

    x_skip = x.clone()
    x = self.layer_norm_2(x)
    x = self.feed_forward(x)
    x = x + x_skip
    x = self.dropout_2(x)

    return x

class TransformerDecoder(nn.Module):
  def __init__(self, model_emb, num_heads, hidden, dropout_p, num_layers):
    super(TransformerDecoder, self).__init__()
    self.decoder = DecoderSequential(*[DecoderBlock(model_emb, num_heads, hidden, dropout_p) 
                                       for _ in range(num_layers)])

  def forward(self, x, mask = None):
    return self.decoder(x, mask)