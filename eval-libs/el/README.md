# Entity Linking Evaluation Library

This evaluation library is heavily based on the [SympTEMIST Official Evaluation Library](https://github.com/nlp4bia-bsc/symptemist_evaluation_library). The only differences are:
- We changed the `symptemist_evaluation.py` file name to `evaluate.py`.
- We created the `build_el_test_file_reference_tsvs.py` file, which creates the Gold Standard TSVs (the ones provided were not in the correct structure).
- We removed the score computations for the NER and CODING subtasks, and only left the methods that compute the accuracy obtained in the EL subtask.
- We rewrote the `utils.write_results` method to simply log that accuracy in a JSON file.

The key point is: **we did not alter any method related to the EL accuracy computation in any way**.

## Usage Instructions

This program is called automatically when using the EL Pipeline (in `scripts/el/sapbert/pipeline`) to train and evaluate the models, and that's how it should be used. Nevertheless, this program compares two TSV files, with one being the reference file (i.e. Gold Standard data provided by the task organizers) and the other being the predictions or results file. This last TSV file needs to have the following structure:

- filename, label, start_span, end_span, text, code

Then, run the library from the terminal using the following command:

```commandline
python3 symptemist_evaluation.py [-d symptemist] [-l es] -r symptemist_el_ref.tsv -p symptemist_el_pred.tsv -o results.json
```

These are the possible arguments:

+ ```-d/--dataset```: name of the dataset. OPTIONAL: should only be used in the EL Pipeline to build the Gold Standard TSV when the path passed in `-r` is invalid (does not exist).
+ ```-l/--language```: language of the dataset. OPTIONAL: should only be used in the EL Pipeline to build the Gold Standard TSV when the path passed in `-r` is invalid (does not exist).
+ ```-r/--reference```: path to Gold Standard TSV file with the annotations.
+ ```-p/--prediction```: path to predictions TSV file with the annotations.
+ ```-o/--output_file```: path of the scoring results file.

## Citation

@inproceedings{symptemist,
  author       = {Lima-L{\'o}pez, Salvador and Farr{\'e}-Maduell, Eul{\`a}lia and Gasco-S{\'a}nchez, Luis and Rodr{\'i}guez-Miret, Jan and Krallinger, Martin},
  title        = {{Overview of SympTEMIST at BioCreative VIII: Corpus, Guidelines and Evaluation of Systems for the Detection and Normalization of Symptoms, Signs and Findings from Text}},
  booktitle    = {Proceedings of the BioCreative VIII Challenge and Workshop: Curation and Evaluation in the era of Generative Models},
  year         = 2023
}
