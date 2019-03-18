import argparse
import os
import pickle
import torch.utils

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.dataset import PreprocessedDataset
    from utils.training import train_single_input_classifier_ff as train_model
    from utils.testing import test_single_input_classifier_ff as test_model
    from stacked_cnn import SCNN
else:
    from utils.dataset import PreprocessedDataset
    from utils.training import train_single_input_classifier_ff as train_model
    from utils.testing import test_single_input_classifier_ff as test_model
    from .stacked_cnn import SCNN

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ModelID', help='Model id. If none, I will generate one based on the system time')
parser.add_argument('--Tag_size', help='how many target classes. need to match the preprocessed dataset', type=int, default=7)
parser.add_argument('--Datapath', help='the base path of the datasets', default='../')
parser.add_argument('--Dataset', help='folder name of the processed data', default='SampleData')
parser.add_argument('--Gpu', help='gpu id to use', default=None)

parser.add_argument('--Batch_size', default=64, type=int)
parser.add_argument('--Emb_dim', default=256, type=int)
parser.add_argument('--Hidden_dim', default=256, type=int)
parser.add_argument('--Kernel_size', default=3, type=int, help='width of the convolutional kernels')
parser.add_argument('--Pooling_size', default=2, type=int)
parser.add_argument('--Num_blocks', default=2, type=int, help='number of convolutional blocks. each block has two convolutional kernels and a pooling layer')
parser.add_argument('--Sentence_length', default=64, type=int, help='max sentence length. need to match the preprocessed dataset')
parser.add_argument('--Workers', default=0, type=int)
parser.add_argument('--Val_split', default=0.1, type=float)
parser.add_argument('--Early_stop_patience', default=5, type=int)
parser.add_argument('--Epochs', default=30, type=int)

args=parser.parse_args()

if args.Gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.Gpu

batch_size = args.Batch_size
emb_dim = args.Emb_dim
hidden_dim = args.Hidden_dim
max_length = args.Sentence_length
kernel_size = args.Kernel_size
pooling_size = args.Pooling_size
num_blocks = args.Num_blocks
Workers = args.Workers
Val_split = args.Val_split
Early_stop_patience = args.Early_stop_patience

if args.ModelID is not None:
    train_id = args.ModelID
else:
    train_id = None

data_dir = args.Datapath
data_sets = os.listdir(data_dir)
data_set = args.Dataset
print('Current Data Set:', data_set)

work_path = os.path.join(data_dir, data_set)
tag_size = args.Tag_size
epochs = args.Epochs

vocab_dump_file = os.path.join(work_path, 'vocab.pkl')
vocab = pickle.load(open(vocab_dump_file, 'rb'))
vocab_size = len(vocab)
padding_idx = vocab.index('<pad>')

train_path = os.path.join(work_path, 'train')
test_path = os.path.join(work_path, 'test')

model = SCNN(vocab_size, emb_dim, hidden_dim, max_length, tag_size, padding_idx, kernel_size=kernel_size,
                 pooling_size=pooling_size, num_blocks=num_blocks)
train_set = PreprocessedDataset(os.path.join(train_path, 'tokens.idx'), os.path.join(train_path, 'tags.idx'))
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=Workers,
)
train_id, best_tmp_path, epoch_num = train_model(model, train_loader, train_set.__len__(),
                                                 batch_size, Val_split, Early_stop_patience, train_path,
                                                 epochs, train_id=train_id)

test_set = PreprocessedDataset(os.path.join(test_path, 'tokens.idx'), os.path.join(test_path, 'tags.idx'))
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=Workers,
)
# resume the best parameters
test_model(train_id, model, test_loader, batch_size, test_set.__len__(), tag_size, test_path)
