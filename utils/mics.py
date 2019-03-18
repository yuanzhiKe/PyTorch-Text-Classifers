import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def pad_sequence(sequence, max_length, end_symbol, use_end_symbol, pad=None):
    assert isinstance(use_end_symbol, bool)
    if pad is None:
        pad = '<pad>'
    if isinstance(sequence, list):
        # pad the radical sequence to the same size
        if len(sequence) < max_length:
            if use_end_symbol:
                if len(sequence) > 0 and sequence[-1] != end_symbol:
                    sequence.append(end_symbol)
            sequence += [pad] * (max_length - len(sequence))
        elif len(sequence) > max_length:
            if use_end_symbol:
                sequence = sequence[:max_length - 1]
                sequence += [end_symbol]
            else:
                sequence = sequence[:max_length]
        else:
            if use_end_symbol:
                if len(sequence) > 0 and sequence[-1] != end_symbol:
                    sequence[-1] = end_symbol
        return sequence
    else:
        raise Exception("Function pad_character expects list.")


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
        np_mask = np_mask.cuda()
    return np_mask


def create_mask(src, padding_idx):
    src_mask = (src != padding_idx).unsqueeze(-2)
    return src_mask


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])