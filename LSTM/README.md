## The LSTM classifier
The LSTM classifier.  
Supports Bidirectional/Unidirectional LSTM and Self-attention over LSTM outputs

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
                         [--Sentence_length SENTENCE_LENGTH]
                         [--Workers WORKERS] [--Val_split VAL_SPLIT]
                         [--Early_stop_patience EARLY_STOP_PATIENCE]
                         [--Epochs EPOCHS] [-dir DIRECTION]
                         [-den ADDTIONAL_DENSE] [-relu RELU] [-att ATTENTION]

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
  --Sentence_length SENTENCE_LENGTH
                        max sentence length. need to match the preprocessed
                        dataset
  --Workers WORKERS
  --Val_split VAL_SPLIT
  --Early_stop_patience EARLY_STOP_PATIENCE
  --Epochs EPOCHS
  -dir DIRECTION, --direction DIRECTION
                        0 or 1 for unidirectional, other for bidirection.
                        Default is 2.
  -den ADDTIONAL_DENSE, --addtional_dense ADDTIONAL_DENSE
                        0 for no addional dense layer, other for more. Default
                        is 0.
  -relu RELU, --relu RELU
                        0 for no relu, other for relu. Default is 1.
  -att ATTENTION, --attention ATTENTION
                        0 for no attention, other for perform a self-attention
                        over the lstm outputs. Default is 0.
```