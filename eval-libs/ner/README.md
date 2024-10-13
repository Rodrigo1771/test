# Named Entity Recognition Evaluation Library

This evaluation library is heavily based on the [MultiCardioNER Official Evaluation Library](https://github.com/nlp4bia-bsc/multicardioner_evaluation_library). The only differences are:
- We changed the `multicardioner_evaluation.py` file name to `evaluate.py`.
- We added ability to compute some additional statistics for better assessment of the model's performance.
- We rewrote the `utils.write_results` method to simply log that accuracy in a JSON file.

The key point is: **we did not alter the main logic related to the NER metrics computation in any way**.

## Usage Instructions

This program is called automatically when using the NER Pipeline (in `scripts/ner/pipeline`) to train and evaluate the models, and that's how it should be used. Nevertheless, this program compares two TSV files, with one being the reference file (i.e. Gold Standard) and the other being the predictions or results file. The TSV files need to have the following structure:

- filename, label, start_span, end_span, text

Then, run the library from the terminal using the following command:

```commandline
python3 evaluate.py -r ner_ref.tsv -p ner_pred.tsv -o results.json
```

These are the possible arguments:

+ ```-r/--reference```: path to Gold Standard TSV file with the annotations (obtained by running `scripts/ner/phrase-parse/testset_to_phrases_parse.py`).
+ ```-p/--prediction```: path to predictions TSV file with the annotations.
+ ```-o/--output_file```: path of the scoring results file.

## Citation

@inproceedings{multicardioner2024overview, title={{Overview of MultiCardioNER task at BioASQ 2024 on Medical Speciality and Language Adaptation of Clinical NER Systems for Spanish, English and Italian}}, author={Salvador Lima-López and Eulàlia Farré-Maduell and Jan Rodríguez-Miret and Miguel Rodríguez-Ortega and Livia Lilli and Jacopo Lenkowicz and Giovanna Ceroni and Jonathan Kossoff and Anoop Shah and Anastasios Nentidis and Anastasia Krithara and Georgios Katsimpras and Georgios Paliouras and Martin Krallinger}, booktitle={CLEF Working Notes}, year={2024}, editor = {Faggioli, Guglielmo and Ferro, Nicola and Galuščáková, Petra and García Seco de Herrera, Alba}}