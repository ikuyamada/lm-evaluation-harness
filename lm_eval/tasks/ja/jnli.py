"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese. 
JGLUE has been constructed from scratch without translation. 

Homepage: https://github.com/yahoojapan/JGLUE
"""
from lm_eval.base import rf
from lm_eval.tasks.xnli import XNLIBase

_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""



class JNLI(XNLIBase):
    VERSION = 1.1
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JNLI"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"]))

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        if doc['label'] == 0:
            label = 0
        elif doc['label'] == 1:
            label = 2
        else:
            label = 1

        return {
            "premise": doc["sentence1"],
            "hypothesis": doc["sentence2"],
            "label": label
        }

    def doc_to_target(self, doc):
        return (
            " "
            + [self.ENTAILMENT_LABEL, self.CONTRADICTION_LABEL, self.NEUTRAL_LABEL][
                doc["label"]
            ]
        )
