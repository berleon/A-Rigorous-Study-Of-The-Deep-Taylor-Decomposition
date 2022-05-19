import dataclasses
import os

import savethat

from relation_network import preprocess


@dataclasses.dataclass(frozen=True)
class PreprocessCLEVRArgs(savethat.Args):
    dataset: str = "../data/clevr/CLEVR_v1.0/"
    only_questions: bool = False


@dataclasses.dataclass(frozen=True)
class PreprocessCLEVRResult:
    pass


class PreprocessCLEVR(
    savethat.Node[PreprocessCLEVRArgs, PreprocessCLEVRResult]
):
    def _run(self):

        if "NLTK_DATA" not in os.environ:
            raise RuntimeError(
                "Please set the environment variable NLTK_DATA "
                "to the location of the NLTK data."
            )
        preprocess.main(self.args.dataset, self.args.only_questions)
        return PreprocessCLEVRResult()
