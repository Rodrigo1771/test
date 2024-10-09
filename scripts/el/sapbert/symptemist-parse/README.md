# SympTEMIST Parse

This directory houses the parsing scripts used to parse the [SympTEMIST](https://temu.bsc.es/symptemist/) dataset for Entity Linking. This dataset is parsed into TXT files with the appropriate format to be fed to the SapBERT pipeline.

## Requirements

The `sapbert_symptemist_parse.py` script expects a folder named `symptemist/` to exist inside the `datasets/` folder in the project's root (so sibling to the `scripts/` folder). Simply [download](https://zenodo.org/records/10635215) the SympTEMIST dataset, rename the main folder to `symptemist/`, and move it to the `datasets/` folder. The final structure should be as follows:

```
─ multilingual-bio-ner-and-el-msc-diss
    ├── datasets
    │   ├── symptemist
    │   │   ├── symptemist_gazetteer
    │   │   │   └── (...)
    │   │   ├── symptemist_multilingual-silver-standard
    │   │   │   └── (...)
    │   │   ├── symptemist_spanish_silver_standard
    │   │   │   └── (...)
    │   │   ├── symptemist_test
    │   │   │   └── (...)
    │   │   └── symptemist_train
    │   │       └── (...)
    │   └── (...)
    ├── scripts
    │   └── (...)
    └── (...)
```

## Output

This process produces four files per language:
- `training_file`: The training file. For performing hyperparameter search, it's composed of 80% of the training data; for training the final model, it is composed of 100% of the training data.
- `validation_file` / `test_file`: For performing hyperparameter search, the validation file is produced, and is ccomposed of the remaining 20% of the training data; for training the final model, the test file is produced.
- `dictionary_file_complete`: The dictionary used for obtaining the candidate codes.
- `dictionary_file_only_test_codes`: A dictionary composed of the `dictionary_file_complete` entries that appear in the `test_file`, solely used as another way of monitoring the validation loss.

The structure of the output directory will be the following:
```
─ out
  ├── hyperparameter-search
  │   ├── en
  │   │   ├── sapbert_symptemist_en_training_file.txt
  │   │   ├── sapbert_symptemist_en_validation_file.txt
  │   │   ├── sapbert_symptemist_en_dictionary_file_complete.txt
  │   │   └── sapbert_symptemist_en_dictionary_file_only_test_code.txt
  │   ├── es
  │   │   └── (...)
  │   ├── fr
  │   │   └── (...)
  │   ├── it
  │   │   └── (...)
  │   └── pt
  │       └── (...)
  └── final-model
      ├── en
      │   ├── sapbert_symptemist_en_training_file.txt
      │   ├── sapbert_symptemist_en_test_file.txt
      │   ├── sapbert_symptemist_en_dictionary_file_complete.txt
      │   └── sapbert_symptemist_en_dictionary_file_only_test_code.txt
      ├── es
      │   └── (...)
      ├── fr
      │   └── (...)
      ├── it
      │   └── (...)
      └── pt
          └── (...)
```

NOTE: The script actually produces another file (per language), containing the Gold Standard anotations in a format suited for the evaluation library. These files are saves in `eval-libs/el/test-file-reference-tsvs/`.