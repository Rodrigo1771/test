# Phrase Parse

This directory houses the script that parses the test splits of each dataset into JSON files with the following structure:

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

## Requirements

The parsing assumes that each dataset is in a folder called `datasets` in the project's root directory (sibling to the `scripts` folder):

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
