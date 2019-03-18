## The Transformer classifier
The classifier which employs Mult-head Self-attention Encoder.  
As it is only for text classification, it does not contain the decoder.  

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
                         [--Sentence_length SENTENCE_LENGTH]
                         [--Workers WORKERS] [--Val_split VAL_SPLIT]
                         [--Early_stop_patience EARLY_STOP_PATIENCE]
                         [--Epochs EPOCHS] [--Heads HEADS]
                         [--N_layers N_LAYERS] [--Multi_gpu MULTI_GPU]

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
  --Sentence_length SENTENCE_LENGTH
                        max sentence length. need to match the preprocessed
                        dataset
  --Workers WORKERS
  --Val_split VAL_SPLIT
  --Early_stop_patience EARLY_STOP_PATIENCE
  --Epochs EPOCHS
  --Heads HEADS
  --N_layers N_LAYERS
  --Multi_gpu MULTI_GPU
```

### Reference
 Vaswani, et al. [Attention is All You Need](https://arxiv.org/abs/1706.03762).
