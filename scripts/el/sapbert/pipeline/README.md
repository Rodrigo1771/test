# The Entity Linking Pipeline

This directory houses the Entity Linking pipeline. It uses the SapBERT training pipeline to train the models. This pipeline is housed in the `models/sapbert` directory (in the project's root).

## Requirements

The pipeline expects the [SympTEMIST](https://temu.bsc.es/symptemist/) dataset to already be parsed: see the [README](../symptemist-parse/README.md) on parsing SympTEMIST for EL. Furthermore, it expects the same dataset to already be augmented: see the [README](../../../data-aug/augment/README.md) on augmenting SympTEMIST for EL. Finally, in order to run the pipeline, the ability to upload the fine-tuned models to your Hugging Face hub is needed. To accomplish this, fill in the `scripts/config` file with the following information:

- Your Hugging Face username (hf_username)
- Your Hugging Face Token (hf_token)

## Usage Instructions

The pipeline can either be executed locally, or on the [DI Cluster](https://cluster.di.fct.unl.pt), in three phases:
- Hyperparameter search (`sapbert_hyperparameter_search.sh`)
- Training the final model (`sapbert_train_and_evaluate_final_model.sh`)
- Training the final model with the augmented datasets (`sapbert_train-aug_and_evaluate_final_model.sh`)

### Locally

Run the pipeline from the terminal using one of the following commands:

```commandline
./sapbert_hyperparameter_search.sh DATASET LANG
./sapbert_train_and_evaluate_final_model.sh DATASET LANG
./sapbert_train-aug_and_evaluate_final_model.sh DATASET LANG MODEL_TYPE
```

These are the possible arguments:

- `DATASET`: Possible values are `symptemist`.
- `LANG`: Possible values are `en`, `es`, `fr`, `it` and `pt` (the languages we used for SympTEMIST).
- `MODEL_TYPE`: Possible values are `word2vec` and `fasttext`.

The possible argument combinations are:
```commandline
./sapbert_hyperparameter_search.sh symptemist en
./sapbert_hyperparameter_search.sh symptemist es
./sapbert_hyperparameter_search.sh symptemist fr
./sapbert_hyperparameter_search.sh symptemist it
./sapbert_hyperparameter_search.sh symptemist pt
./sapbert_train_and_evaluate_final_model.sh symptemist en
./sapbert_train_and_evaluate_final_model.sh symptemist es
./sapbert_train_and_evaluate_final_model.sh symptemist fr
./sapbert_train_and_evaluate_final_model.sh symptemist it
./sapbert_train_and_evaluate_final_model.sh symptemist pt
./sapbert_train-aug_and_evaluate_final_model.sh symptemist en word2vec
./sapbert_train-aug_and_evaluate_final_model.sh symptemist en fasttext
./sapbert_train-aug_and_evaluate_final_model.sh symptemist es word2vec
./sapbert_train-aug_and_evaluate_final_model.sh symptemist es fasttext
./sapbert_train-aug_and_evaluate_final_model.sh symptemist fr word2vec
./sapbert_train-aug_and_evaluate_final_model.sh symptemist fr fasttext
./sapbert_train-aug_and_evaluate_final_model.sh symptemist it word2vec
./sapbert_train-aug_and_evaluate_final_model.sh symptemist it fasttext
./sapbert_train-aug_and_evaluate_final_model.sh symptemist pt word2vec
./sapbert_train-aug_and_evaluate_final_model.sh symptemist pt fasttext
```

The result files will be saved in `out/{DATASET}/hyperparameter-search/{LANG}` for hyperparameter search, `out/{DATASET}/final-model/{LANG}` for training the final model, and `out/{DATASET}/final-model-aug/{MODEL_TYPE}/{LANG}` for training the final model with the augmented datasets.

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

- `DATASET`: Possible values are `symptemist`.
- `LANGS`: Possible values are `en`, `es`, `fr`, `it` and `pt` (the languages we used for SympTEMIST).
- `MODE`: Possible values are `hyperparameter-search`, `final-model`, `final-mode-aug`.
- `MODEL_TYPE`: Possible values are `word2vec` and `fasttext`.
- `ALLOCATION_TIME`: Possible values are any amount of hours you'd like.

The possible variable combinations are:
- `DATASET`="symptemist" ; `LANGS`=("en" "es" "fr" "it" "pt") ; `MODE`="hyperparameter-search" ; `MODEL_TYPE`="";
- `DATASET`="symptemist" ; `LANGS`=("en" "es" "fr" "it" "pt") ; `MODE`="final-model" ; `MODEL_TYPE`="";
- `DATASET`="symptemist" ; `LANGS`=("en" "es" "fr" "it" "pt") ; `MODE`="final-model" ; `MODEL_TYPE`="word2vec";
- `DATASET`="symptemist" ; `LANGS`=("en" "es" "fr" "it" "pt") ; `MODE`="final-model" ; `MODEL_TYPE`="fasttext";

The `node_allocator.sh` script allocates as many cluster nodes as there are languages in the LANGS variable. It then passes to each node the `container_handler.sh` script with the variables outlined earlier. The `container_handler.sh` script then launches a Docker container that executes either the `sapbert_hyperparameter_search.sh`, the `sapbert_train_and_evaluate_final_model.sh` or the `sapbert_train-aug_and_evaluate_final_model.sh` script depending on the value of the `MODE` variable. Finally, when the training and evaluation is complete, the `container_handler.sh` script extracts the results file from the Docker container and saves it in `your-new-folder/out/{DATASET}/{MODE}/{MODEL_TYPE}/{LANG}/` in the cluster (`MODEL_TYPE` can be "" in which case that dir will be skipped). You can copy them to `scripts/el/sapbert/pipeline/out/{TASK}/out/{DATASET}/{MODE}/{MODEL_TYPE}/{LANG}/` in your local machine (again, `MODEL_TYPE` can be "").