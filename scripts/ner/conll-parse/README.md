# CONLL Parse

This directory houses the scripts used to parse the [SympTEMIST](https://temu.bsc.es/symptemist/), [CANTEMIST](https://temu.bsc.es/cantemist/) and [MultiCardioNER](https://temu.bsc.es/multicardioner/) datasets for Named Entity Recognition. These datasets are parsed into CONLL files to be uploaded to Hugging Face, in order to be used in the [Datasets](https://huggingface.co/docs/datasets/index) library.

NOTE: The files in each dataset are handwritten free-text reports and don't have a well-defined structure. Thus, this code, beyond being _research code_, is very specific to each dataset, i.e. very hardcoded and very spaghetti. Furthermore, for lack of time or simply for not being relevant enough at the time, a group of very few dataset specific errors were not corrected.

## Requirements

The parsing expects each dataset to be in a folder called `datasets/` in the project's root (so sibling to the `scripts/` folder). Simply **download** the [SympTEMIST](https://zenodo.org/records/10635215), [CANTEMIST](https://zenodo.org/records/3978041),  [MultiCardioNER](https://zenodo.org/records/11368861) and [MedProcNER](https://zenodo.org/records/8224056) datasets, rename the folders to `symptemist/`, `cantemist/`, `multicardioner/` and `medprocner/` respectively, and move them to the `datasets/` folder. The final structure should be as follows:

```
─ multilingual-bio-ner-and-el-msc-diss
    ├── datasets
    │    ├── cantemist
    │    │    └── (...)
    │    ├── medprocner
    │    │    └── (...)
    │    ├── multicardioner
    │    │    └── (...)
    │    └── symptemist
    │         └── (...)
    ├── scripts
    │    └── (...)
    └── (...)
```

## General Subfolder Structure

Each subfolder represents the parsing of a different dataset. They have the following structure:

```
─ <DATASET>-parse
     ├── out
     │   ├── <SPLIT>.conll
     │   └── (...)
     ├── <DATASET>_to_conll_parse.py
     └── <DATASET>_loading_script.py
```

- `out/`: The output folder.
  - `<SPLIT>.conll`: The CONLL file that corresponds to a split (train, dev, test).
- `<DATASET>_to_conll_parse.py`: The file that parses the dataset. It takes as input the paths where each splits' directory is, and the split itself (the `mode` var). These inputs are given in the script itself, as variables. 
- `<DATASET>_loading_script.py`: The loading script for the HF repo. Adapted from [this script](https://huggingface.co/datasets/PlanTL-GOB-ES/cantemist-ner/blob/main/cantemist-ner.py).

NOTE: The parsing of MultiCardioNER depends on the `symptemist-parse/out/train-full.conll` and `medprocner-parse/out/train-full.conll` files, so parse SympTEMIST and MedProcNER first.


### Dataset Specific Quirks

- `symptemist-parse/symptemist_to_conll_parse.py`: Parsing SympTEMIST results in CONLL files for the train, dev, and test splits as normal. However, it also results in another file, which corresponds to all splits together. This file is named `out/train-full.conll` and is used by the `multicardioner-parse/multicardioner_to_conll_parse.py` script to build the training files for the MultiCardioNER experiments with the 4 entity types (check the MultiCardioNER system description paper [here](https://ceur-ws.org/Vol-3740/paper-11.pdf) for more info on these experiments). To generate this file, you need to create a new directory named `train+test/`, and then copy the contents of the `symptemist/train/subtask1-ner/brat/` and `symptemist/test/subtask1-ner/brat/` directories to this new directory.
- `medprocner-parse/medprocner_to_conll_parse.py`: Only outputs the `out/train-full.conll` file described earlier, and for the same reason. To generate this file, the same steps apply (changing `symptemist` for `medprocner` of course).
- `multicardioner-parse/multicardioner_to_conll_parse.py`: This script uses an auxiliary JSON file `selected_examples_idxs.json` to select the training and validation examples, for all spanish `train.conll` and `dev.conll` files, respectively. This is so the splits are the same as the ones used in the dissertation, and not random every time.

## Uploading to Hugging Face

When all datasets are parsed, upload them to Hugging Face with the `upload_datasets_to_huggingface.py` script before running the NER pipeline. This script requires credentials in `scripts/config`.
