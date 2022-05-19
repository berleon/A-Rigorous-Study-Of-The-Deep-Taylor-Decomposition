import dataclasses
import pickle
from pathlib import Path

import savethat

from relation_network import preprocess


@dataclasses.dataclass(frozen=True)
class PreprocessCLEVR_XAIArgs(savethat.Args):
    clevr: str = "../data/CLEVR_v1.0/"
    clevr_xai: str = "../data/CLEVR_XAI_v1.0/"


class PreprocessCLEVR_XAI(savethat.Node[PreprocessCLEVR_XAIArgs, None]):
    def _run(self) -> None:
        clevr = Path(self.args.clevr)
        clevr_xai = Path(self.args.clevr_xai)

        questions: list[str] = [
            "CLEVR-XAI_complex_questions.json",
            "CLEVR-XAI_simple_questions.json",
        ]
        with open(clevr / "dic.pkl", "rb") as f:
            encoding = pickle.load(f)

        for question_file in questions:
            print(f"Processing {question_file}")
            preprocess.process_question(
                root=None,
                split=None,
                word_dic=encoding["word_dic"],
                answer_dic=encoding["answer_dic"],
                question_file=clevr_xai / question_file,
                result_file=(clevr_xai / question_file).with_suffix(".pkl"),
            )

        print("Processing images")
        preprocess.process_image(
            clevr_xai / "images",
            clevr_xai / "images_preprocessed",
        )
