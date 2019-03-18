## The Paralledl CNN classifier
The CNN whose blocks are arranged in one layer proposed by [Kim 2014]  

![Alt text](image.png?raw=true "Parallel CNN")

## How to run the demo.
### Quick Start
Running

    python train_and_test.py
    
will automatically learn the sample data and give the test report.

### Options
```angular2
usage: train_and_test.py [-h] [-i MODELID] [--Tag_size TAG_SIZE]
                         [--Datapath DATAPATH] [--Dataset DATASET] [--Gpu GPU]
                         [--Batch_size BATCH_SIZE] [--Emb_dim EMB_DIM]
                         [--Hidden_dim HIDDEN_DIM]
                         [--Kernel_sizes KERNEL_SIZES [KERNEL_SIZES ...]]
                         [--Pooling_size POOLING_SIZE]
                         [--Sentence_length SENTENCE_LENGTH]
                         [--Workers WORKERS] [--Val_split VAL_SPLIT]
                         [--Early_stop_patience EARLY_STOP_PATIENCE]
                         [--Epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -i MODELID, --ModelID MODELID
                        Model id. If none, I will generate one based on the
                        system time
  --Tag_size TAG_SIZE   how many target classes. need to match the
                        preprocessed dataset
  --Datapath DATAPATH   the base path of the datasets
  --Dataset DATASET     folder name of the processed data
  --Gpu GPU             gpu id to use
  --Batch_size BATCH_SIZE
  --Emb_dim EMB_DIM
  --Hidden_dim HIDDEN_DIM
  --Kernel_sizes KERNEL_SIZES [KERNEL_SIZES ...]
                        width of the convolutional kernels in each block.
                        Default is [3,4,5]
  --Pooling_size POOLING_SIZE
  --Sentence_length SENTENCE_LENGTH
                        max sentence length. need to match the preprocessed
                        dataset
  --Workers WORKERS
  --Val_split VAL_SPLIT
  --Early_stop_patience EARLY_STOP_PATIENCE
  --Epochs EPOCHS
```

### Reference
 [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014).
