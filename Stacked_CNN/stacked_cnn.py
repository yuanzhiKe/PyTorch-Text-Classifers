import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.mics import get_clones
else:
    from utils.mics import get_clones


class SCNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, hidden_dim, max_length, target_size, padding_idx, kernel_size=3,
                 pooling_size=2, num_blocks=2, pretrained_emb=None):
        super(SCNN, self).__init__()
        if pretrained_emb is None:
            self.emb_lookup = nn.Embedding(vocab_size, emb_dim, scale_grad_by_freq=False, padding_idx=padding_idx)
        else:
            print("Read Pretrained Embs.")
            self.emb_lookup = nn.Embedding.from_pretrained(pretrained_emb)

        self.num_blocks = num_blocks
        self.cnn1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=emb_dim, out_channels=hidden_dim, kernel_size=kernel_size)),
            ('conv2', nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size)),
            ('activation', torch.nn.ReLU()),
            ('maxpooling', nn.MaxPool1d(kernel_size=pooling_size))
        ]))
        length = (max_length - kernel_size - 1) // pooling_size
        if num_blocks>1:
            self.cnn_blocks = get_clones(nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size)),
                ('conv2', nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size)),
                ('activation', torch.nn.ReLU()),
                ('maxpooling', nn.MaxPool1d(kernel_size=pooling_size))
            ])), num_blocks-1)
            for _ in range(num_blocks-1):
                length = (length - kernel_size - 1) // pooling_size
        self.hidden2tag = nn.Linear(length*hidden_dim, target_size)

    def forward(self, x):
        """
        forward
        :param x: [batch, input_len]
        :return: [batch, target_size]
        """
        embedding_representations = self.emb_lookup(x)
        embedding_representations = torch.transpose(embedding_representations, 1, 2)
        out = self.cnn1(embedding_representations)
        for i in range(self.num_blocks-1):
            out = self.cnn_blocks[i](out)
        out = out.view(x.size(0), -1)
        tags = self.hidden2tag(out)
        return tags

    def predict(self, x):
        tags = self.forward(x)
        return F.softmax(tags, 1)

