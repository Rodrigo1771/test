# Loading script for the CombinedTrainDisTEMISTDevNER dataset.
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
"""


_DESCRIPTION = """\
"""

_URL = "https://huggingface.co/datasets/Rodrigo1771/combined-train-distemist-dev-ner/resolve/main/"
_TRAINING_FILE = "train.conll"
_DEV_FILE = "dev.conll"
_TEST_FILE = "test.conll"


class CombinedTrainDisTEMISTDevNERConfig(datasets.BuilderConfig):
    """BuilderConfig for CombinedTrainDisTEMISTDevNER NER dataset"""

    def __init__(self, **kwargs):
        """BuilderConfig for CombinedTrainDisTEMISTDevNER NER.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CombinedTrainDisTEMISTDevNERConfig, self).__init__(**kwargs)


class CombinedTrainDisTEMISTDevNER(datasets.GeneratorBasedBuilder):
    """CombinedTrainDisTEMISTDevNER NER dataset."""

    BUILDER_CONFIGS = [
        CombinedTrainDisTEMISTDevNERConfig(
            name="CombinedTrainDisTEMISTDevNER",
            version=datasets.Version("1.0.0"),
            description="CombinedTrainDisTEMISTDevNER dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-ENFERMEDAD",
                                "I-ENFERMEDAD",
                                "B-PROCEDIMIENTO",
                                "I-PROCEDIMIENTO",
                                "B-SINTOMA",
                                "I-SINTOMA",
                                "B-FARMACO",
                                "I-FARMACO"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_DESCRIPTION,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # CombinedTrainDisTEMISTDevNER tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[-1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
