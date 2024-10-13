# The Named Entity Recognition Pipeline

This directory houses the Named Entity Recognition pipeline. It uses the SapBERT training pipeline to train the models. This pipeline is housed in the `models/sapbert` directory (in the project's root).

## Requirements

The pipeline expects the [SympTEMIST](https://temu.bsc.es/symptemist/), [CANTEMIST](https://temu.bsc.es/cantemist/) and [MultiCardioNER](https://temu.bsc.es/multicardioner/) datasets to already be parsed and uploaded to Hugging Face: see this [README](../conll-parse/README.md) and this [README](../phrase-parse/README.md) on parsing these datasets for NER. Furthermore, it expects the same datasets to already be augmented: see the [README](../../data-aug/augment/README.md) on augmenting these datasets for NER. Finally, in order to run the pipeline, the ability to upload the fine-tuned models to your Hugging Face hub is needed. To accomplish this, fill in the `scripts/config` file with the following information:

- Your Hugging Face username (hf_username)
- Your Hugging Face Token (hf_token)

## Usage Instructions

The pipeline can either be executed locally, or on the [DI Cluster](https://cluster.di.fct.unl.pt), in two phases:
- Training the final model (`ner_train_and_evaluate.sh`)
- Training the final model with the augmented datasets (`ner_train-aug_and_evaluate.sh`)

### Locally

Run the pipeline from the terminal using one of the following commands:

```commandline
./ner_train_and_evaluate.sh DATASET MODEL
./ner_train-aug_and_evaluate.sh DATASET MODEL MODEL_TYPE
```

These are the possible arguments:

- `DATASETS`: Possible values are `symptemist`, `cantemist`, `distemist`, `drugtemist-es`, `drugtemist-en` and `drugtemist-it`.
- `MODELS`: Possible values are `PlanTL-GOB-ES/bsc-bio-ehr-es` for the datasets `symptemist`, `cantemist`, `distemist` and `drugtemist-es`; `michiyasunaga/BioLinkBERT-base` for `drugtemist-en`, and `IVN-RIN/bioBIT` for `drugtemist-it`.
- `MODEL_TYPE`: Possible values are `word2vec` and `fasttext`.

The possible argument combinations are:
```commandline
./ner_train_and_evaluate.sh symptemist PlanTL-GOB-ES/bsc-bio-ehr-es
./ner_train_and_evaluate.sh cantemist PlanTL-GOB-ES/bsc-bio-ehr-es
./ner_train_and_evaluate.sh distemist PlanTL-GOB-ES/bsc-bio-ehr-es
./ner_train_and_evaluate.sh drugtemist-es PlanTL-GOB-ES/bsc-bio-ehr-es
./ner_train_and_evaluate.sh drugtemist-en michiyasunaga/BioLinkBERT-base
./ner_train_and_evaluate.sh drugtemist-it IVN-RIN/bioBIT
./ner_train-aug_and_evaluate.sh symptemist PlanTL-GOB-ES/bsc-bio-ehr-es word2vec
./ner_train-aug_and_evaluate.sh symptemist PlanTL-GOB-ES/bsc-bio-ehr-es fasttext
./ner_train-aug_and_evaluate.sh cantemist PlanTL-GOB-ES/bsc-bio-ehr-es word2vec
./ner_train-aug_and_evaluate.sh cantemist PlanTL-GOB-ES/bsc-bio-ehr-es fasttext
./ner_train-aug_and_evaluate.sh distemist PlanTL-GOB-ES/bsc-bio-ehr-es word2vec
./ner_train-aug_and_evaluate.sh distemist PlanTL-GOB-ES/bsc-bio-ehr-es fasttext
./ner_train-aug_and_evaluate.sh drugtemist-es PlanTL-GOB-ES/bsc-bio-ehr-es word2vec
./ner_train-aug_and_evaluate.sh drugtemist-es PlanTL-GOB-ES/bsc-bio-ehr-es fasttext
./ner_train-aug_and_evaluate.sh drugtemist-en michiyasunaga/BioLinkBERT-base word2vec
./ner_train-aug_and_evaluate.sh drugtemist-en michiyasunaga/BioLinkBERT-base fasttext
./ner_train-aug_and_evaluate.sh drugtemist-it IVN-RIN/bioBIT word2vec
./ner_train-aug_and_evaluate.sh drugtemist-it IVN-RIN/bioBIT fasttext
```

The result files will be saved in `out/`, under `*_final_results.json` where "*" is the dataset and model (and model type and distance threshold if training with augmented datasets).

### DI Cluster

Run the pipeline from the DI Cluster by first copying the contents of the `dicluster/` folder to the cluster. Put it all inside a folder of your choosing. Its structure should look like:

```
─ your-new-folder
    ├── container_handler.sh
    ├── Dockerfile
    └── node_allocator.sh
```

Then, execute the node allocator like:

```commandline
./node_allocator.sh
```

This script contains five variables:

- `DATASETS`: Possible values are `symptemist`, `cantemist`, `distemist`, `drugtemist-es`, `drugtemist-en` and `drugtemist-it`.
- `MODELS`: Possible values are `PlanTL-GOB-ES/bsc-bio-ehr-es` for the datasets `symptemist`, `cantemist`, `distemist` and `drugtemist-es`; `michiyasunaga/BioLinkBERT-base` for `drugtemist-en`, and `IVN-RIN/bioBIT` for `drugtemist-it`.
- `MODEL_TYPE`: Possible values are `word2vec` and `fasttext`.
- `ALLOCATION_TIME`: Possible values are any amount of hours you'd like.

The possible variable combinations are:
- `DATASETS`=("symptemist" "cantemist" "distemist" "drugtemist-es" "drugtemist-en" "drugtemist-it") ; `MODELS`=("PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "michiyasunaga/BioLinkBERT-base" "IVN-RIN/bioBIT") ; `MODEL_TYPE`="";
- `DATASETS`=("symptemist" "cantemist" "distemist" "drugtemist-es" "drugtemist-en" "drugtemist-it") ; `MODELS`=("PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "michiyasunaga/BioLinkBERT-base" "IVN-RIN/bioBIT") ; `MODEL_TYPE`="word2vec";
- `DATASETS`=("symptemist" "cantemist" "distemist" "drugtemist-es" "drugtemist-en" "drugtemist-it") ; `MODELS`=("PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "PlanTL-GOB-ES/bsc-bio-ehr-es" "michiyasunaga/BioLinkBERT-base" "IVN-RIN/bioBIT") ; `MODEL_TYPE`="fasttext";

The `node_allocator.sh` script allocates as many cluster nodes as there are datasets in the DATASETS variable (and models in the MODELS variable, cus they have to have the same number of values). It then passes to each node the `container_handler.sh` script with the variables outlined earlier. The `container_handler.sh` script then launches a Docker container that executes either the `ner_train_and_evaluate.sh`, or the `ner_train-aug_and_evaluate.sh` script depending on the value of the `MODEL_TYPE` variable. Finally, when the training and evaluation is complete, the `container_handler.sh` script extracts the results file from the Docker container and saves it in `your-new-folder/out/` in the cluster. You can copy them to `scripts/ner/pipeline/out/` in your local machine.