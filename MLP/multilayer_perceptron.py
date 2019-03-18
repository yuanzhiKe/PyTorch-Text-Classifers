import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.mics import get_clones
else:
    from utils.mics import get_clones


class MLP(nn.Module):
    def __init__(self,vocab_size, emb_dim, target_size, n_layers=3, hidden_dim=None, padding_idx=None, pretrained_emb=None):
        super(MLP, self).__init__()
        if pretrained_emb is None:
            self.emb_lookup = nn.Embedding(vocab_size, emb_dim, scale_grad_by_freq=False, padding_idx=padding_idx)
        else:
            print("Read Pretrained Embs.")
            self.emb_lookup = nn.Embedding.from_pretrained(pretrained_emb)
        if hidden_dim is None:
            hidden_dim = emb_dim
        self.n_layers = n_layers
        if n_layers < 3:
            raise ValueError('n_layers should be at least 3')
        self.emb2hidden = nn.Linear(emb_dim, hidden_dim)
        if n_layers > 3:
            self.map_layers = get_clones(nn.Linear(hidden_dim, hidden_dim), n_layers-3)
        else:
            self.map_layers = None
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        x = self.emb_lookup(x)
        x = torch.sum(x, 1)
        x = self.emb2hidden(x)
        if self.map_layers is not None:
            for i in range(self.n_layers):
                x = self.map_layers[i](x)
        out = self.hidden2tag(x)
        return out

    def predict(self, x):
        x = self.emb_lookup(x)
        x = torch.sum(x, 1)
        x = self.emb2hidden(x)
        if self.map_layers is not None:
            for i in range(self.n_layers):
                x = self.map_layers[i](x)
        out = self.hidden2tag(x)
        return F.softmax(out, 1)

