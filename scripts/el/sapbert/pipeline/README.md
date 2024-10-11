# The Entity Linking Pipeline

This directory houses the parsing scripts used to parse the [SympTEMIST](https://temu.bsc.es/symptemist/) dataset for Entity Linking. This dataset is parsed into TXT files with the appropriate format to be fed to the SapBERT pipeline (check `models/sapbert/README.md` and the [SapBERT repository](https://github.com/cambridgeltl/sapbert) for more information on the pipeline).

- sapbert in "models" dir
- dataset in "datasets" dir
- running it in the di cluster vs running it locally
- remember config

- test by downloading all the datasets, and put the evaluation libraries not in the datasets folder, but in a evaluation folder 