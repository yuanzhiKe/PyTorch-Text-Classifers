## The fastText classifier
The unigram implementation.

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
  --Workers WORKERS
  --Val_split VAL_SPLIT
  --Early_stop_patience EARLY_STOP_PATIENCE
  --Epochs EPOCHS
```

## References

A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@InProceedings{joulin2017bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month={April},
  year={2017},
  publisher={Association for Computational Linguistics},
  pages={427--431},
}
```