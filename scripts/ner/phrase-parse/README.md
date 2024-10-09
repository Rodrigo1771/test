# Phrase Parse

This directory houses the script that parses the test splits of each dataset, phrase by phrase.

## Requirements

The parsing expects each dataset to be in a folder called `datasets/` in the project's root (so sibling to the `scripts/` folder). Simply **download** the [SympTEMIST](https://zenodo.org/records/10635215), [CANTEMIST](https://zenodo.org/records/3978041) and [MultiCardioNER](https://zenodo.org/records/11368861) datasets, rename the folders to `symptemist/`, `cantemist/` and `multicardioner/` respectively, and move them to the `datasets/` folder (basically the same thing as the CONLL parsing requirements). The final structure should be as follows:

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

## Output

The `testsets_to_phrases_parse.py` script will produce JSON files with the following structure:

```json
[
    [
        <TEXT>,
        <FILE_NAME>,
        <START_SPAN>
    ],
    ...
]
```

- `<TEXT>`: A phrase present in the given dataset.
- `<FILE_NAME>`: The name of the dataset's file from which that phrase was taken.
- `<START_SPAN>`: The start span of that phrase inside the document (i.e. the index of the first character).

These JSON files will be used to obtain the predicted entities (with the `run_ner_predict_into_tsv.py` script), phrase by phrase, and then adjust their spans to the whole document.

