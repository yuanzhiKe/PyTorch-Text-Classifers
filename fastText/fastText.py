import torch
import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):
    def __init__(self, vocab_size, emb_dim, target_size, padding_idx=None, pretrained_emb=None):
        super(FastText, self).__init__()
        if pretrained_emb is None:
            self.emb_lookup = nn.Embedding(vocab_size, emb_dim, scale_grad_by_freq=False, padding_idx=padding_idx)
        else:
            print("Read Pretrained Embs.")
            self.emb_lookup = nn.Embedding.from_pretrained(pretrained_emb)
        self.hidden2tag = nn.Linear(emb_dim, target_size)

    def forward(self, x):
        """
        forward
        :param x: [batch, input_len]
        :return: [batch, target_size]
        """
        embedding_representations = self.emb_lookup(x)
        out = torch.sum(embedding_representations, 1)
        return self.hidden2tag(out)

    def predict(self, x):
        out = self.forward(x)
        return F.softmax(out, 1)
