# Data Augmentation Pipeline

This directory houses the scripts used to augment the NER and EL data.

## Requirements

### 1. Train the Word2Vec models 

Follow the instructions in `scripts/data-aug/train-word2vec/README.md`. The trained models should be saved in `scripts/data-aug/train-word2vec/out/{LANG}/` (with LANG being the language of the model).

### 2. Download the pre-trained FastText models

Download the pre-trained FastText models [here](https://fasttext.cc/docs/en/crawl-vectors.html#models). Dowload the `text` versions and save them in `scripts/data-aug/fasttext-models/` (no need to uncompress them).

### 3. Parse the NER datasets 

Follow the instructions in `scripts/ner/conll-parse/README.md`.

### 4. Parse the EL SympTEMIST datasets

Follow the instructions in `scripts/el/symptemist-parse/README.md`.

## Usage Instructions

The pipeline can either be executed locally, or on the [DI Cluster](https://cluster.di.fct.unl.pt).

### Locally

Run the pipeline from the terminal using one of the following commands (the first for NER, inside the `ner` folder, and the second for EL, inside the `el` folder):

```commandline
./augment_ner_data.sh DATASET MODEL_TYPE
./augment_el_data.sh LANG MODEL_TYPE
```

These are the possible arguments:

- `DATASET`: Only for NER. Possible values are `symptemist`, `cantemist` and `multicardioner`.
- `LANG`: Only for EL. Possible values are `en`, `es`, `fr`, `it` and `pt` (the languages we used for SympTEMIST).
- `MODEL_TYPE`: For both NER and EL. Possible values are `word2vec` and `fasttext`.
- (other similarity thresholds can be used by modifying the `DISTANCE_THRESHOLDS` variables inside those two scripts)

The possible argument combinations are:
```commandline
./augment_ner_data.sh symptemist word2vec
./augment_ner_data.sh cantemist word2vec
./augment_ner_data.sh multicardioner word2vec
./augment_ner_data.sh symptemist fasttext
./augment_ner_data.sh cantemist fasttext
./augment_ner_data.sh multicardioner fasttext
./augment_el_data.sh en word2vec
./augment_el_data.sh en fasttext
./augment_el_data.sh es word2vec
./augment_el_data.sh es fasttext
./augment_el_data.sh fr word2vec
./augment_el_data.sh fr fasttext
./augment_el_data.sh it word2vec
./augment_el_data.sh it fasttext
./augment_el_data.sh pt word2vec
./augment_el_data.sh pt fasttext
```

The augmented datasets will be saved in `ner/out/{MODEL_TYPE}/{DATASET}/` for NER and `el/out/{MODEL_TYPE}/{LANG}/` for EL, and can be used in the NER and EL pipelines.

### DI Cluster

Run the pipeline from the DI Cluster by first copying the following files and folders to the cluster (put it all inside a folder of your choosing):
- The contents of the `dicluster/` folder. 
- The `scripts/config` file with your credentials.
- The contents of the `scripts/data-aug/train-word2vec/out/` folder into a folder of name `models/word2vec/`.
- The contents of the `scripts/data-aug/fasttext-models/` folder into a folder of name `models/fasttext/`.

The structure of this new folder should look like:

```
─ your-new-folder
    ├── config
    ├── container_handler.sh
    ├── Dockerfile
    ├── node_allocator.sh
    └── models
        ├── word2vec
        │   └── (...)
        └── fasttext
            └── (...)
```

Then, execute the node allocator like:

```commandline
./node_allocator.sh
```

This script contains four variables (`GITHUB_TOKEN` should not be modified):

- `TASK`: Possible values are `ner` and `el`.
- `ARGS`: Represents the arguments for either the NER script or the EL script: if `TASK`=`ner`, possible values are `symptemist`, `cantemist` and `multicardioner`; if `TASK`=`el`, possible values are `en`, `es`, `fr`, `it` and `pt`.
- `MODEL_TYPE`: Possible values are `word2vec` and `fasttext`.
- `ALLOCATION_TIME`: Possible values are any amount of hours you'd like.

The possible variable combinations are:
- `TASK`="ner" ; `ARGS`=("cantemist" "symptemist" "multicardioner") ; `MODEL_TYPE`="word2vec";
- `TASK`="ner" ; `ARGS`=("cantemist" "symptemist" "multicardioner") ; `MODEL_TYPE`="fasttext";
- `TASK`="el" ; `ARGS`=("en" "es" "fr" "it" "pt") ; `MODEL_TYPE`="word2vec";
- `TASK`="el" ; `ARGS`=("en" "es" "fr" "it" "pt") ; `MODEL_TYPE`="fasttext";

The `node_allocator.sh` script allocates as many cluster nodes as there are arguments in the ARGS variable. It then passes to each node the `container_handler.sh` script with the variables outlined earlier. The `container_handler.sh` script then launches a Docker container that executes either the `augment_ner_data.sh` or the ``augment_el_data.sh`` script depending on the value of the `TASK` variable. Finally, when the augmentation is complete, the `container_handler.sh` script extracts the augmented files from the Docker container and saves them in `your-new-folder/out/{TASK}/{MODEL_TYPE}/{ARG}/` in the cluster. Copy them to `scripts/data-aug/augment/{TASK}/out/{MODEL_TYPE}/{ARG}/` in your local machine, to use them in the NER and EL pipelines.
