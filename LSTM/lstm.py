import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.mics import get_clones
    from utils.attention import MultiHeadAttention
else:
    from utils.mics import get_clones
    from utils.attention import MultiHeadAttention

class LSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim, hidden_size, bidirectional, addional_dense, use_cuda=True,
                 use_attention=False, relu=False, pretrained_emb=None):
        super(LSTM, self).__init__()
        if pretrained_emb is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        else:
            print("Read Pretrained Embs.")
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb)
        self.lstm = torch.nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, bidirectional=bidirectional)
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.relu = relu
        if bidirectional:
            output_size = 2 * hidden_size
        else:
            output_size = hidden_size
        self.addtional_dense = addional_dense
        if addional_dense == 1 and not relu:
            self.h2t = nn.Sequential(nn.Linear(output_size, 2 * output_size), nn.Linear(2 * output_size, n_classes))
        elif addional_dense == 1 and relu:
            self.h2t = nn.Sequential(nn.Linear(output_size, 2 * output_size), nn.ReLU(), nn.Linear(2 * output_size, n_classes))
        elif addional_dense > 1 and not relu:
            self.lstm2h = nn.Linear(output_size, 2 * output_size)
            self.denselayers = get_clones(nn.Linear(2 * output_size, 2 * output_size), addional_dense - 1)
            self.h2t = nn.Linear(2 * output_size, n_classes)
        elif addional_dense > 1 and relu:
            self.lstm2h = nn.Sequential(nn.Linear(output_size, 2 * output_size), nn.ReLU())
            self.denselayers = get_clones(nn.Linear(2 * output_size, 2 * output_size), addional_dense - 1)
            self.h2t = nn.Linear(2 * output_size, n_classes)
        else:
            self.h2t = nn.Linear(output_size, n_classes)
        if use_attention:
            self.att = MultiHeadAttention(heads=1, d_model=output_size)

    def init_hidden(self, batch_size, hidden_size, direction):
        if self.use_cuda:
            return (torch.autograd.Variable(
                torch.cuda.FloatTensor(1 * direction, batch_size, hidden_size).fill_(0)),
                    torch.autograd.Variable(
                        torch.cuda.FloatTensor(1 * direction, batch_size, hidden_size).fill_(
                            0)))
        else:
            return (
            torch.autograd.Variable(torch.zeros(1 * direction, batch_size, hidden_size)),
            torch.autograd.Variable(torch.zeros(1 * direction, batch_size, hidden_size)))

    def forward(self, x, hidden=None, length=None):
        x = self.embedding(x)
        x_lstm = x.transpose(0, 1)
        max_time, batch_size, _ = x_lstm.size()
        if length is None:
            length = torch.autograd.Variable(torch.LongTensor([max_time] * batch_size))
            if x_lstm.is_cuda:
                device = x_lstm.get_device()
                length = length.cuda(device)
        if self.bidirectional:
            direction = 2
        else:
            direction = 1
        if hidden is None:
            hidden = self.init_hidden(batch_size, self.hidden_size, direction)
        output, hidden = self.lstm(x_lstm, hidden)
        if self.use_attention:
            output = output.transpose(0, 1)
            output = self.att(output, output, output)
            output = torch.sum(output, 1)
        elif not self.bidirectional:
            output = output[-1]
        else:
            output = output.transpose(0, 1)
            output = torch.sum(output, 1)
        if self.addtional_dense > 1:
            pred = self.lstm2h(output)
            for i in range(self.addtional_dense - 2):
                pred = self.denselayers[i](pred)
            pred = self.h2t(pred)
        else:
            pred = self.h2t(output)
        return pred

    def predict(self, x):
        pred = self.forward(x)
        return F.softmax(pred, 1)

