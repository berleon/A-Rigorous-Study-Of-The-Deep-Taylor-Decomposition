"""Evaluate explanation technique on the CLEVR XAI dataset.

This module computes the saliency maps for the relation network
and evaluates how well the explanation technique matches the
ground truth heatmaps.
"""

# from lrp_relations import enable_deterministic  # noqa isort:skip

import dataclasses
import pickle
from typing import Optional, cast

import numpy as np
import pandas as pd
import savethat
import torch
from savethat import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from lrp_relations import data, lrp, train_clevr
from relation_network import model as rel_model


@dataclasses.dataclass(frozen=True)
class GroundTruthEvalArgs(savethat.Args):
    model_key: str
    dataset: str = "../data/clevr/CLEVR_v1.0/"
    question_type: str = "simple"  # "simple" or "complex"
    ground_truth: str = "single_object"  # "single_object" or "all_objects"
    n_samples: int = -1  # -1 for all samples
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 50
    checkpoint: str = "best"


@dataclasses.dataclass(frozen=True)
class GroundTruthEvalResults:
    relevance_mass: torch.Tensor
    relevance_rank_accuracy: torch.Tensor
    correct: torch.Tensor

    def as_dataframe(self) -> pd.DataFrame:
        rank = self.relevance_rank_accuracy.cpu().numpy()
        return pd.DataFrame(
            {
                "relevance_mass": self.relevance_mass.cpu().numpy(),
                "relevance_rank_accuracy": rank,
            }
        )

    def accuracy(self) -> float:
        return self.correct.float().mean().item()


def load_model(
    storage: savethat.Storage,
    key: str,
    checkpoint: str = "best",
    map_location: Optional[torch.device] = None,
) -> tuple[rel_model.RelationNetworks, train_clevr.TrainArgs]:
    """Load the model from the storage.

    Args:
        storage: Storage to load the model from.
        key: Key of the model to load.

    Returns:
        The model.
    """
    if not (storage / key).exists():
        storage.download(key)

    if checkpoint == "best":
        with open(storage / key / "results.pickle", "rb") as f:
            results = cast(train_clevr.TrainedModel, pickle.load(f))
        ckpt = results.checkpoints[-1]
        ckpt_path = storage / key / "checkpoints" / ckpt.path
        logger.debug(
            f"Loading model with accuracy {ckpt.accuracy:.4f} from {ckpt_path}"
        )
    else:
        ckpt_path = storage / key / "checkpoints" / checkpoint
        logger.debug(f"Loading model from {ckpt_path}")

    model = rel_model.RelationNetworks(data.get_n_words())
    model.load_state_dict(torch.load(ckpt_path, map_location=map_location))

    args = train_clevr.TrainArgs.from_json(storage / key / "args.json")
    return model, args


def relevance_mass(
    saliency: torch.Tensor,
    mask: torch.Tensor,
    reduce: tuple[int, ...] = (1, 2, 3),
) -> torch.Tensor:
    """Compute the relevance mass.

    Args:
        saliency: Saliency map.
        mask: Mask to apply.
        reduce: Dimensions to reduce.

    Returns:
        The relevance mass.
    """
    within = (saliency * mask).sum(dim=reduce)
    total = saliency.sum(dim=reduce)
    return within / total


def l2_norm_sq(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Compute the L2 norm squared.

    Args:
        x: Tensor to compute the L2 norm squared.

    Returns:
        The L2 norm squared.
    """
    return (x**2).sum(dim, keepdim=True)


def max_norm(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Compute the max norm.

    Args:
        x: Tensor to compute the max norm.
        dim: Dimension to compute the max norm.

    Returns:
        The max norm.
    """
    max, _ = x.abs().max(dim, keepdim=True)
    return max


# -----------------------------------------------------------------------------
# Functions for computing the relevance rank accuracy
# copied from IBA code ;)


def to_index_map(hmap: np.ndarray) -> np.ndarray:
    """Return a heatmap, in which every pixel has its value-index as value"""
    order_map = np.zeros_like(hmap, dtype=np.int64)
    for i, idx in enumerate(to_index_list(hmap)):
        order_map[idx] = -i
    return order_map


def to_index_list(
    hmap: np.ndarray, reverse: bool = False
) -> list[tuple[np.ndarray]]:
    """Return the list of indices that would sort this map.

    Sorting order is highest pixel first, lowest last
    """
    order = np.argsort((hmap if reverse else -hmap).ravel())
    indices = np.unravel_index(order, hmap.shape)  # array of two tuples
    indices_trans = np.transpose(np.stack(indices))
    return [tuple(i) for i in np.stack(indices_trans)]  # type: ignore


def get_ration_in_mask(heatmap: np.ndarray, mask: np.ndarray) -> float:
    if mask.ndim != 2:
        raise ValueError("Expected 2 dimensions")
    if heatmap.ndim != 2:
        raise ValueError("Expected 2 dimensions")
    if mask.shape != heatmap.shape:
        raise ValueError("Shapes must match")

    heatmap_idxs = to_index_map(heatmap).astype(np.int64)
    mask_np = mask > 0.5
    heatmap_bbox_idxs = heatmap_idxs.copy()
    heatmap_bbox_idxs[mask_np == 0] = heatmap_idxs.min()
    n_pixel_in_mask = mask_np.sum()
    return float(
        (heatmap_bbox_idxs > (-n_pixel_in_mask)).sum() / n_pixel_in_mask.sum()
    )


class GroundTruthEval(
    savethat.Node[GroundTruthEvalArgs, GroundTruthEvalResults]
):
    def _run(self):
        device = torch.device(self.args.device)

        model, model_args = load_model(
            self.storage,
            self.args.model_key,
            self.args.checkpoint,
            map_location=device,
        )
        model.to(device)
        lrp_model = lrp.LRPViewOfRelationNetwork(model)
        lrp_model.to(device)

        dataset = data.CLEVR_XAI(
            self.args.dataset,
            self.args.question_type,
            self.args.ground_truth,
            model_args.reverse_question,
            use_preprocessed=True,
        )

        if self.args.n_samples == -1:
            n_samples = len(dataset)
        else:
            n_samples = self.args.n_samples

        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            collate_fn=dataset.collate_data,
        )

        # mse = 0
        pbar = tqdm(loader)
        rel_mass = []
        rel_rank = []

        total_samples = 0
        correct = []
        for i, (image, question, q_len, answer, gt) in enumerate(pbar):
            if i > n_samples // self.args.batch_size:
                break

            image, question, answer, gt = (
                image.to(device),
                question.to(device),
                answer.to(device),
                gt.to(device),
            )

            saliency, logits = lrp_model.get_lrp_saliency_and_logits(
                image,
                question,
                q_len,
                target=answer,
                normalize=False,
            )

            correct.append((logits.argmax(1) == answer).cpu())

            rel_mass.append(
                relevance_mass(l2_norm_sq(saliency), gt).detach().cpu()
            )
            ranks = [
                get_ration_in_mask(
                    max_norm(s, dim=0).cpu().detach().numpy()[0],
                    gt_mask.cpu().detach().numpy()[0],
                )
                for gt_mask, s in zip(gt, saliency)
            ]
            rel_rank.append(torch.tensor(ranks).cpu())
            total_samples += len(image)

        res = GroundTruthEvalResults(
            relevance_mass=torch.cat(rel_mass),
            relevance_rank_accuracy=torch.cat(rel_rank),
            correct=torch.cat(correct),
        )
        print("-" * 80)
        print(f"Statistics on {total_samples} samples:")
        print(res.as_dataframe().describe())
        print("-" * 80)
        print(f"Accuracy: {res.accuracy():.4f}")
        print("-" * 80)
        return res
