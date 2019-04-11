import torch
from torch.utils import data
from .mics import create_mask, is_number
import random


class PreprocessedDataset(data.Dataset):

    def __init__(self, input_file, tag_file, data_limit=None):
        if data_limit is None:
            self.input_lines = [line for line in open(input_file)]
            self.tag_lines = [line for line in open(tag_file)]
        else:
            self.input_lines = [line for line in open(input_file)][:data_limit]
            self.tag_lines = [line for line in open(tag_file)][:data_limit]

    def __getitem__(self, index):
        input_line = self.input_lines[index]
        tag_line = self.tag_lines[index]
        input_line = input_line.replace('\n', '')
        tag_line = tag_line.replace('\n', '')
        input_tensor = torch.LongTensor([int(x) for x in input_line.split(' ') if is_number(x)])
        tag_tensor = torch.LongTensor([int(x) for x in tag_line.split(' ') if is_number(x)])
        return input_tensor, tag_tensor

    def __len__(self):
        return len(self.input_lines)


class PreprocessedDataset_mask(data.Dataset):

    def __init__(self, input_file, tag_file, padding_idx, data_limit=None):
        if data_limit is None:
            self.input_lines = [line for line in open(input_file)]
            self.tag_lines = [line for line in open(tag_file)]
        else:
            self.input_lines = [line for line in open(input_file)][:data_limit]
            self.tag_lines = [line for line in open(tag_file)][:data_limit]
        self.padding_idx = padding_idx

    def __getitem__(self, index):
        input_line = self.input_lines[index]
        tag_line = self.tag_lines[index]
        input_line = input_line.replace('\n', '')
        tag_line = tag_line.replace('\n', '')
        input_tensor = torch.LongTensor([int(x) for x in input_line.split(' ') if is_number(x)])
        tag_tensor = torch.LongTensor([int(x) for x in tag_line.split(' ') if is_number(x)])
        mask_tensor = create_mask(input_tensor, self.padding_idx)
        return input_tensor, tag_tensor, mask_tensor

    def __len__(self):
        return len(self.input_lines)
