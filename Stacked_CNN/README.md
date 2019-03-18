## The Stacked CNN classifier
The stacked CNN (which often used for image processing).  
The convolutional blocks are stacked one by one like the follows:

![Alt text](image.png?raw=true "Stacked CNN")

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
                         [--Hidden_dim HIDDEN_DIM] [--Kernel_size KERNEL_SIZE]
                         [--Pooling_size POOLING_SIZE]
                         [--Num_blocks NUM_BLOCKS]
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
  --Kernel_size KERNEL_SIZE
                        width of the convolutional kernels
  --Pooling_size POOLING_SIZE
  --Num_blocks NUM_BLOCKS
                        number of convolutional blocks. each block has two
                        convolutional kernels and a pooling layer
  --Sentence_length SENTENCE_LENGTH
                        max sentence length. need to match the preprocessed
                        dataset
  --Workers WORKERS
  --Val_split VAL_SPLIT
  --Early_stop_patience EARLY_STOP_PATIENCE
  --Epochs EPOCHS
```