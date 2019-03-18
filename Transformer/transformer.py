"""
Reference:
https://arxiv.org/abs/1706.03762
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from torch.autograd import Variable

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.mics import get_clones
    from utils.attention import MultiHeadAttention
else:
    from utils.mics import get_clones
    from utils.attention import MultiHeadAttention


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x, training=True):
        if training:
            x = self.dropout(F.relu(self.linear_1(x)))
        else:
            x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask, training=True):
        x2 = self.norm_1(x)
        if training:
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        else:
            x = x + self.attn(x2, x2, x2, mask)
        x2 = self.norm_2(x)
        if training:
            x = x + self.dropout_2(self.ff(x2, training))
        else:
            x = x + self.ff(x2, training)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, N, heads, dropout, pretrained_emb = None, **kwargs):
        super().__init__()
        self.N = N
        if pretrained_emb is None:
            self.embed = nn.Embedding(vocab_size, emb_dim)
        else:
            print("Read Pretrained Embs.")
            self.embed = nn.Sequential(nn.Embedding.from_pretrained(pretrained_emb), nn.Linear(300, emb_dim))
        self.pe = PositionalEncoder(emb_dim, dropout=dropout, **kwargs)
        self.layers = get_clones(EncoderBlock(emb_dim, heads, dropout), N)
        self.norm = Norm(emb_dim)

    def forward(self, src, mask, training=True):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask, training)
        return self.norm(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_classes, padding_idx, max_seq_length, n_layers=6, heads=8, dropout=0.1, pretrained_emb=None, **kwargs):
        super(TransformerClassifier, self).__init__()
        self.transformer = TransformerEncoder(vocab_size, emb_dim, n_layers, heads, dropout, max_seq_len=max_seq_length, pretrained_emb=pretrained_emb)
        self.h2t = nn.Linear(emb_dim, n_classes)
        self.padding_idx = padding_idx

    def forward(self, x, mask, training):
        h = self.transformer(x, mask, training=training)
        h_sum = torch.sum(h, 1)  # sum as the sentence vector
        pred = self.h2t(h_sum)
        return pred

    def predict(self, x, mask):
        h = self.transformer(x, mask, training=False)
        h_sum = torch.sum(h, 1)  # sum as the sentence vector
        pred = self.h2t(h_sum)
        return F.softmax(pred, 1)
