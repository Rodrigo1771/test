# Wor2Vec Model Training Pipeline

This directory houses the scripts used to train a Word2Vec model, for any language. It's heavily adapted from the [Kyubyong/wordvectors](https://github.com/Kyubyong/wordvectors) project.

The pipeline (represented by the `training_pipeline.sh` file) encompasses the following four steps, for a given language:
- Downloads the [Wikimedia Database Backup Dumps](https://dumps.wikimedia.org) for that language.
- Uncompresses the downloaded data.
- Builds a corpus of every sentence from that data.
- Trains a model on that corpus.

## Usage Instructions

The pipeline can either be executed locally, or on the [DI Cluster](https://cluster.di.fct.unl.pt). 

### Locally

Run the pipeline from the terminal using the following command, with LANG being the language to train the model on (it can be any language supported by the Wikimedia Database Backup Dumps, but in this dissertation the languages used were "en", "es", "fr", "it" and "pt"):

```commandline
./training_pipeline.sh LANG
```

The trained model will be saved in `out/{LANG}`.

### DI Cluster

Run the pipeline from the DI Cluster by copying to the cluster the contents of the `dicluster` folder, plus the `config` file with your credentials, and executing the node allocator like:

```commandline
./node_allocator.sh
```

This script allocates as many cluster nodes as there are languages in `node_allocator.sh`'s LANG variable. It then passes to each node the `container_handler.sh` script with the respective language as attribute. The `container_handler.sh` script then launches a Docker container that executes the `training_pipeline.sh` script from earlier, again with the respective language as attribute. Finally, when the training is complete, the `container_handler.sh` script extracts the trained model from the Docker container and saves it in `out/{LANG}` (in the cluster, you can then copy it to your local machine).
