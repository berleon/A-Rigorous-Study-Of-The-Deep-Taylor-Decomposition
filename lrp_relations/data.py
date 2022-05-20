import pickle
from pathlib import Path
from typing import Optional, Union

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

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, np.ndarray, int, int, torch.Tensor]:

        question_idx, gt_path = self.gt_filenames[index]
        imgfile, question, answer, _ = self.data[question_idx]

        img_path = {
            False: self.root / "images" / imgfile,
            True: self.root / "images_preprocessed" / imgfile,
        }[self.use_preprocessed]

        img = Image.open(img_path).convert("RGB")
        if not self.use_preprocessed:
            img = rel_data.resize(img)

        # always use eval transform
        img = rel_data.eval_transform(img)

        if self.reverse_question:
            question = question[::-1]

        gt = torch.from_numpy(np.load(gt_path)).float()
        return img, question, len(question), answer, gt

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_data(
        batch: list[tuple[torch.Tensor, np.ndarray, int, int, torch.Tensor]]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor,
        torch.Tensor,
    ]:
        images, lengths, answers, gt_masks = [], [], [], []
        batch_size = len(batch)

        max_len = max(map(lambda x: len(x[1]), batch))

        questions = np.zeros((batch_size, max_len), dtype=np.int64)
        sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        for i, b in enumerate(sort_by_len):
            image, question, length, answer, gt = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
            gt_masks.append(gt)

        return (
            torch.stack(images),
            torch.from_numpy(questions),
            lengths,
            torch.LongTensor(answers),
            torch.stack(gt_masks).unsqueeze(1),
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
    )
    return loader
