# Entity Recognition and Linking for Multilingual Biomedical Documents

This repository contains the code from our [dissertation]() on Named Entity Recognition (NER) and Entity Linking (EL) for the biomedical domain in multilingual scenarios. It implements both our transfer learning and data augmentation approaches to tackle those two tasks.

## Requirements

### Packages

This project depends on the following python packages (also available in the `requirements.txt` file, install them with `pip`):

- tqdm 
- humanize 
- gensim 
- regex 
- lxml 
- numpy
- torch 
- pandas 
- pytorch_metric_learning 
- transformers 
- sentencepiece 
- protobuf

### Utilities

It also depends on the following utilities (install them with your preferred package manager):

- python3
- wget 
- bzip2 
- pv

### Datasets and Shared Tasks

The experiments were carried out on the [SympTEMIST](https://temu.bsc.es/symptemist/), [CANTEMIST](https://temu.bsc.es/cantemist/) and [MultiCardioNER](https://temu.bsc.es/multicardioner/) shared tasks. Thus, to replicate those experiments, follow these steps:

1. Download those datasets: [SympTEMIST](https://zenodo.org/records/10635215), [CANTEMIST](https://zenodo.org/records/3978041) and [MultiCardioNER](https://zenodo.org/records/11368861).
2. Additionally, to replicate the MultiCardioNER experiments outlined in our [system description paper](https://ceur-ws.org/Vol-3740/paper-11.pdf), download [MedProcNER](https://zenodo.org/records/8224056).
3. Change the folder names to `symptemist`, `cantemist`, `multicardioner` and `medprocner`, respectively.
4. Place them in a new folder named `datasets` in the root of the repository.

The final structure of the root directory should look like this:
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
    ├── eval-libs
    │    └── (...)
    ├── models
    │    └── (...)
    └── scripts
         └── (...)
```

### Config

In order to run the pipelines, the ability to upload the fine-tuned models to your Hugging Face hub is needed. To accomplish this, fill in the `scripts/config` file with the following information:

- Your Hugging Face username (hf_username)
- Your Hugging Face Token (hf_token)

## Project Structure

The project follows the following structure:

```
─ multilingual-bio-ner-and-el-msc-diss
    ├── eval-libs
    │    ├── el
    │    │    └── (...)
    │    └── ner
    │         └── (...)
    ├── models
    │    └── sapbert
    │         └── (...)
    └── scripts
         ├── data-aug
         │    ├── augment
         │    │    └── (...)
         │    └── train-word2vec
         │         └── (...)
         ├── el
         │    └── sapbert
         │         ├── pipeline
         │         │    └── (...)
         │         └── symptemist-parse
         │              └── (...)
         ├── ner
         │    ├── conll-parse
         │    │    └── (...)
         │    ├── phrase-parse
         │    │    └── (...)
         │    └── pipeline
         │         └── (...)
         ├── utils
         │    └── (...)
         └── config
```

Each folder has a different purpose (from top to bottom):

- `eval-libs`: The official evaluation libraries used.
  - `el`: The EL evaluation library, adapted from the [SympTEMIST official evaluation library](https://github.com/nlp4bia-bsc/symptemist_evaluation_library) (check this [README](eval-libs/el/README.md)).
  - `ner`: The NER evaluation library, adapted from the [MultiCardioNER official evaluation library](https://github.com/nlp4bia-bsc/multicardioner_evaluation_library) (check this [README](eval-libs/ner/README.md)).
- `models`: The models we used that need to be local because they envolve a larger pipeline, not just the model.
  - `sapbert`: The SapBERT Pipeline, adapted from [here](https://github.com/cambridgeltl/sapbert) (check this [README](models/sapbert/README.md)).
- `scripts`: The code used to perform our NER and EL transfer learning and data augmentation experiments.
  - `data-aug`: Data augmentation code.
    - `augment`: The code to augment the NER and EL datasets (check this [README](scripts/data-aug/augment/README.md)).
    - `train-word2vec`: The code to train our own Word2Vec models (check this [README](scripts/data-aug/train-word2vec/README.md)).
  - `el`: The code for the EL transfer learning experiments.
    - `sapbert`: (we only used the sapbert pipeline, but more can be added).
      - `pipeline`: The EL pipeline code (check this [README](scripts/el/sapbert/pipeline/README.md)).
      - `symptemist-parse`: The code to parse the SympTEMIST EL dataset (check this [README](scripts/el/sapbert/symptemist-parse/README.md)).
  - `ner`: The code for the NER transfer learning experiments.
    - `conll-parse`: The code to parse the NER datasets (check this [README](scripts/ner/conll-parse/README.md)).
    - `phrase-parse`: The code to parse the NER datasets' testsets (check this [README](scripts/ner/phrase-parse/README.md)).
    - `pipeline`: The NER pipeline code (check this [README](scripts/ner/pipeline/README.md)).
  - `utils`: Files common to both the NER and EL pipelines (currently just the `upload_model_to_huggingface.py` file).
  - `config`: Config file with Hugging Face Credentials.

## Project Flow

As previously said, this project deals with NER and EL for the biomedical domain in multilingual scenarios. In our experiments, we used transfer learning and data augmentation techniques to tackle those two tasks, and utilized the aforementioned datasets. Thus, these experiments can be divided into:

1. NER Transfer Learning Experiments
2. NER Data Augmentation Experiments
3. EL Transfer Learning Experiments
4. EL Data Augmentation Experiments

To replicate each one of these experiments, follow these steps (in each of the enumerated paths there's a complementary README with more information on how to use that specific code):

1. NER Transfer Learning Experiments
   1. Parse the NER datasets in `scripts/ner/conll-parse/` ([README](scripts/ner/conll-parse/README.md)) and in `scripts/ner/phrase-parse/` ([README](scripts/ner/phrase-parse/README.md)).
   2. Run the NER pipeline in `scripts/ner/pipeline/` ([README](scripts/ner/pipeline/README.md)).
2. NER Data Augmentation Experiments
   1. Train the Word2Vec word embedding models in `scripts/data-aug/train-word2vec/` ([README](scripts/data-aug/train-word2vec/README.md)).
   2. Download the FastText word embedding models [here](https://fasttext.cc/docs/en/crawl-vectors.html#models).
   3. Augment the NER datasets in `scripts/data-aug/augment/ner` ([README](scripts/data-aug/augment/README.md)).
   4. Run the NER pipeline again in `scripts/ner/pipeline/` ([README](scripts/ner/pipeline/README.md)).
3. EL Transfer Learning Experiments
   1. Parse the SympTEMIST EL dataset in `scripts/el/sapbert/symptemist-parse/` ([README](scripts/el/sapbert/symptemist-parse/README.md)).
   2. Run the EL pipeline in `scripts/el/sapbert/pipeline/` ([README](scripts/el/sapbert/pipeline/README.md)).
4. EL Data Augmentation Experiments
   1. Steps i. and ii. of "2. NER Data Augmentation Experiments"
   2. Augment the SympTEMIST EL dataset in `scripts/data-aug/augment/el` ([README](scripts/data-aug/augment/README.md)).
   3. Run the EL pipeline again in `scripts/el/sapbert/pipeline/` ([README](scripts/el/sapbert/pipeline/README.md)).

## Citation

TODO