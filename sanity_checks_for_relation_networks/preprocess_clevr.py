import dataclasses

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
        preprocess.main(self.args.dataset, self.args.only_questions)
        return PreprocessCLEVRResult()
