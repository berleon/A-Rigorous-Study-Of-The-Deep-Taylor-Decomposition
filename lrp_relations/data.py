import json
import pickle
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import torch
import torch.utils.data as torch_data
from PIL import Image

import relation_network.dataset as rel_data
from lrp_relations import utils


def get_n_words() -> int:
    data_root = utils.clevr_path()
    with open(data_root / "dic.pkl", "rb") as f:
        dic = pickle.load(f)
    n_words = len(dic["word_dic"]) + 1
    return n_words


def get_clevr_dataloader(
    data_root: Path,
    split: str = "train",
    reverse_question: bool = True,
    batch_size: int = 30,
    n_worker: int = 8,
    shuffle: Optional[bool] = None,
) -> torch_data.DataLoader:
    shuffle = shuffle if shuffle is not None else split == "train"
    return torch_data.DataLoader(
        rel_data.CLEVR(
            data_root,
            split,
            transform=rel_data.transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
        ),
        batch_size=batch_size,
        num_workers=n_worker,
        shuffle=shuffle,
        collate_fn=rel_data.collate_data,
    )


"""

The CLEVR-XAI dataset is organized into the following structure:
-----------------------------------------------------------------------------------------------

./
├── CLEVR-XAI_scenes.json  # CLEVR-XAI scenes, one scene per image
├── CLEVR-XAI_simple_questions.json  # CLEVR-XAI-simple questions
├── CLEVR-XAI_complex_questions.json  # CLEVR-XAI-complex questions
├── images/  # CLEVR-XAI images
│   ├── CLEVR_unique_000000.png
│   ├── CLEVR_unique_000001.png
│   ├── ... [10,000 images in total]
├── masks/  # CLEVR-XAI segmentation masks
│   ├── CLEVR_unique_000000.png
│   ├── CLEVR_unique_000001.png
│   ├── ... [10,000 masks in total]
├── ground_truth_complex_questions_unique/  # GT Unique, on CLEVR-XAI-complex
│   ├── 0.npy
│   ├── 1.npy
│   ├── ... [89,873 ground truths in total]
│   └── stats.json
"""


class CLEVR_XAI(torch_data.Dataset):
    def __init__(
        self,
        root: Union[str, Path, None] = None,
        question_type: str = "simple",
        ground_truth: str = "single_object",
        reverse_question: bool = True,
        use_preprocessed: bool = False,
    ):
        self.root = Path(root if root is not None else utils.clevr_xai_path())
        self.question_type = question_type
        self.ground_truth = ground_truth
        self.use_preprocessed = use_preprocessed
        self.reverse_question = reverse_question

        if self.question_type not in ["simple", "complex"]:
            raise ValueError(
                "question_type must be either 'simple' or 'complex'"
            )
        self.gt_dir = (
            self.root
            / f"ground_truth_{self.question_type}_questions_{self.ground_truth}"
        )
        self.gt_filenames = [
            (int(p.stem), p) for p in sorted(self.gt_dir.glob("*.npy"))
        ]

        questions_pkl = self.root / f"CLEVR-XAI_{question_type}_questions.pkl"
        with open(questions_pkl, "rb") as f:
            self.data = pickle.load(f)

        questions_json = questions_pkl.with_suffix(".json")
        with open(questions_json, "r") as f:
            self.questions_json = json.load(f)["questions"]

    def answer_dict(self) -> dict[int, str]:
        if hasattr(self, "answer_class"):
            return self.answer_class
        root = utils.clevr_path()
        with open(f"{root}/dic.pkl", "rb") as f:
            self.dic = pickle.load(f)
        self.answer_class = {v: k for k, v in self.dic["answer_dic"].items()}
        return self.answer_class

    def get_question_and_answer(self, index: int) -> tuple[str, str]:
        question_idx, _ = self.gt_filenames[index]
        question = self.questions_json[question_idx]["question"]
        answer = self.questions_json[question_idx]["answer"]
        return question, answer

    def get_image(
        self, index: int, preprocessed: bool, resize: bool
    ) -> Image.Image:
        question_idx, _ = self.gt_filenames[index]
        imgfile, _, _, _ = self.data[question_idx]
        img_path = {
            False: self.root / "images" / imgfile,
            True: self.root / "images_preprocessed" / imgfile,
        }[preprocessed]
        img = Image.open(img_path).convert("RGB")

        if resize:
            img = rel_data.resize(img)
        return img

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, np.ndarray, int, int, int]:

        question_idx, _ = self.gt_filenames[index]
        _, question, answer, _ = self.data[question_idx]

        resize = not self.use_preprocessed
        img = self.get_image(index, self.use_preprocessed, resize)

        # always use eval transform
        img_tensor = cast(torch.Tensor, rel_data.eval_transform(img))

        if self.reverse_question:
            question = question[::-1]

        return img_tensor, question, len(question), answer, index

    def get_ground_truth(self, index: int) -> torch.Tensor:
        _, gt_path = self.gt_filenames[index]
        gt = torch.from_numpy(np.load(gt_path)).float()
        return gt

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_data(
        batch: list[tuple[torch.Tensor, np.ndarray, int, int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor, list[int]]:
        images, lengths, answers, indices = [], [], [], []
        batch_size = len(batch)

        max_len = max(map(lambda x: len(x[1]), batch))

        questions = np.zeros((batch_size, max_len), dtype=np.int64)
        sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        for i, b in enumerate(sort_by_len):
            image, question, length, answer, idx = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
            indices.append(idx)

        return (
            torch.stack(images),
            torch.from_numpy(questions),
            lengths,
            torch.LongTensor(answers),
            indices,
        )


def get_clevr_xai_loader(
    root: Union[str, Path, None] = None,
    question_type: str = "simple",
    ground_truth: str = "single_object",
    batch_size: int = 30,
    n_worker: int = 8,
    reverse_question: bool = True,
    use_preprocessed: bool = False,
) -> torch_data.DataLoader:
    dataset = CLEVR_XAI(
        root=root,
        question_type=question_type,
        ground_truth=ground_truth,
        reverse_question=reverse_question,
        use_preprocessed=use_preprocessed,
    )
    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_worker,
        collate_fn=dataset.collate_data,
        shuffle=False,
    )
    return loader
