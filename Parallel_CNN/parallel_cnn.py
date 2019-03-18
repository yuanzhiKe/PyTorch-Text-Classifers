import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.mics import get_clones
else:
    from utils.mics import get_clones


class CNN_block(nn.Module):
    def __init__(self, in_ch, o_ch, kernel_size, pooling_size):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=o_ch, kernel_size=kernel_size)
        self.maxpooling = nn.MaxPool1d(kernel_size=pooling_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.ReLU()(x)
        x = self.maxpooling(x)
        return x


class PCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, max_length, target_size, padding_idx,
                 pooling_size=2, block_sets=None, pretrained_emb=None):
        super(PCNN, self).__init__()
        if pretrained_emb is None:
            self.emb_lookup = nn.Embedding(vocab_size, emb_dim, scale_grad_by_freq=False, padding_idx=padding_idx)
        else:
            print("Read Pretrained Embs.")
            self.emb_lookup = nn.Embedding.from_pretrained(pretrained_emb)
        if block_sets is None:
            block_sets = [3, 4, 5]
        self.block_num = len(block_sets)
        blocks = []
        length = 0
        for kernel_size in block_sets:
            blocks.append(CNN_block(emb_dim, hidden_dim, kernel_size, pooling_size))
            length += (max_length - kernel_size + 1) // pooling_size
        self.blocks = nn.ModuleList(blocks)
        self.hidden2tag = nn.Linear(length*hidden_dim, target_size)

    def forward(self, x):
        """
        forward
        :param x: [batch, input_len]
        :return: [batch, target_size]
        """
        embedding_representations = self.emb_lookup(x)
        embedding_representations = torch.transpose(embedding_representations, 1, 2)
        outs = []
        for i in range(self.block_num):
            out = self.blocks[i](embedding_representations)
            out = out.view(x.size(0), -1)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        tags = self.hidden2tag(out)
        return tags

    def predict(self, x):
        embedding_representations = self.emb_lookup(x)
        embedding_representations = torch.transpose(embedding_representations, 1, 2)
        outs = []
        for i in range(self.block_num):
            out = self.blocks[i](embedding_representations)
            out = out.view(x.size(0), -1)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        tags = self.hidden2tag(out)
        return F.softmax(tags, 1)

