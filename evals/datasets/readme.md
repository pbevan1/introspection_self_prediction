# Datasets

Datasets are defined as .jsonls. Each line in the .jsonl is an entry of the dataset. It needs to minimally contain the following fields:
- `string`: the text of the entry that is used to in the string field.

Additional fields—such as `target`—can also be included and will be written into the experiment output.

The .jsonl should be pointed to from the config file in `conf/task`. This might also contain a prompt that should go with the dataset if the dataset itself requires a wrapper prompt.

Ideally, the dataset is split here into a train and validation part to prevent cross-contamination of the training and validation set down the pipeline. Use `scripts/split_jsonl_into_train_test.py` to do this.

By default, datasets are kept with a precomputed test/validation split computed at creation to be really sure that we don't have leakage.
The naming scheme is `all_<dataset_name>.jsonls` for the full dataset and `train_<dataset_name>.jsonls` and `val_<dataset_name>.jsonls` for the train and validation set, respectively.
Following this scheme is necessary for the default configuration files to work, where a `set` field contains either `all`, `train` or `val` to indicate which part of the dataset is used.
