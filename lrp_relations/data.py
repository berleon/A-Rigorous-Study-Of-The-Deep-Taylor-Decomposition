import pickle
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from lrp_relations import utils
from relation_network.dataset import CLEVR, collate_data, transform


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
) -> DataLoader:
    shuffle = shuffle if shuffle is not None else split == "train"
    return DataLoader(
        CLEVR(
            data_root,
            split,
            transform=transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
        ),
        batch_size=batch_size,
        num_workers=n_worker,
        shuffle=shuffle,
        collate_fn=collate_data,
    )
