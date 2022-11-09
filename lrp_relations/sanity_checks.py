"""Sanity checks on a relation networks.

This script partially reproduces the result from the (Arras et al., 2022) paper.

The main difference is that the relation networks are not trained, i.e. randomly
initialized.

To run the script, execute the following command:
```
$ pkg=sanity_checks_for_relation_networks
$ python -m sanity_checks_for_relation_networks run \
    "$pkg.$pkg.SanityChecksForRelationNetworks" \
    --n_samples 1000
```

"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import pickle
import random
from pathlib import Path
from typing import Callable, Optional, cast

import pandas as pd
import savethat
import torch
from tqdm import tqdm

from lrp_relations import data, lrp, train_clevr, utils
from relation_network import dataset as rel_dataset
from relation_network.model import RelationNetworks

from lrp_relations import enable_deterministic  # noqa isort:skip


@dataclasses.dataclass(frozen=True)
class SanityChecksForRelationNetworksArgs(savethat.Args):
    dataset: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: Optional[str] = None  # the key of the model to use
    checkpoint: Optional[str] = None
    n_samples: int = 1000
    batch_size: int = 100
    question_type: str = "simple"
    ground_truth: str = "single_object"


@dataclasses.dataclass(frozen=True)
class SaliencyResult:
    saliency: torch.Tensor
    images: torch.Tensor
    image_idx: torch.Tensor
    question: torch.Tensor
    question_len: torch.Tensor
    question_index: torch.Tensor
    target: torch.Tensor

    @staticmethod
    def cat(results: list[SaliencyResult]) -> SaliencyResult:
        question_len = torch.cat([r.question_len for r in results], dim=0)
        max_question_len = question_len.max()

        def pad_question(q: torch.Tensor) -> torch.Tensor:
            pad_shape = (q.shape[0], max_question_len - q.shape[1])
            if pad_shape[0] == 0:
                return q
            return torch.cat([q, q.new_zeros(*pad_shape)], dim=1)

        return SaliencyResult(
            saliency=torch.cat([r.saliency for r in results], dim=0),
            images=torch.cat([r.images for r in results], dim=0),
            image_idx=torch.cat([r.image_idx for r in results], dim=0),
            question=torch.cat(
                [pad_question(r.question) for r in results], dim=0
            ),
            question_len=question_len,
            question_index=torch.cat(
                [r.question_index for r in results], dim=0
            ),
            target=torch.cat([r.target for r in results], dim=0),
        )


@dataclasses.dataclass(frozen=True)
class SanityChecksForRelationNetworksResults:
    saliency_0: SaliencyResult
    saliency_1: SaliencyResult
    saliency_0_rand_questions: SaliencyResult

    def statistics(
        self,
        normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> pd.DataFrame:
        def normalize(x: torch.Tensor) -> torch.Tensor:
            if normalize_fn is None:
                return x
            return torch.stack([normalize_fn(xi) for xi in x])

        saliency_0 = normalize(self.saliency_0.saliency)
        saliency_1 = normalize(self.saliency_1.saliency)
        saliency_0_rand_questions = normalize(
            self.saliency_0_rand_questions.saliency
        )

        diff_01 = (saliency_0 - saliency_1).abs()
        diff_question = (saliency_0 - saliency_0_rand_questions).abs()

        items = [
            saliency_0,
            saliency_1,
            saliency_0_rand_questions,
            diff_01,
            diff_question,
        ]

        return pd.DataFrame(
            data=[
                dict(
                    abs_mean=x.abs().mean().item(),
                    abs_std=x.abs().std().item(),
                    min=x.min().item(),
                    max=x.max().item(),
                )
                for x in items
            ],
            index=[
                "LRP_logit_0",
                "LRP_logit_1",
                "LRP_logit_0_permuted_questions",
                "Diff_LRP_logit_01",
                "Diif_LRP_logit_0_permuted_questions",
            ],
        )


def random_answer_same_category_fn(
    dataset: data.CLEVR_XAI,
) -> Callable[[torch.Tensor], torch.Tensor]:
    category_inverse = collections.defaultdict(set)
    for answer_name, category in rel_dataset.category.items():
        category_inverse[category].add(answer_name)

    int_to_answer = dataset.answer_dict()
    answer_to_int = {v: k for k, v in int_to_answer.items()}

    def get_random_answer(answers: torch.Tensor) -> torch.Tensor:
        rand_answers = []
        for i in answers.tolist():
            answer = int_to_answer[i]
            category = rel_dataset.category[answer]
            category_items = category_inverse[category] - set([answer])
            rand_answer = random.choice(list(category_items))
            rand_answers.append(answer_to_int[rand_answer])
        return torch.tensor(rand_answers, device=answers.device)

    return get_random_answer


class SanityChecksForRelationNetworks(
    savethat.Node[
        SanityChecksForRelationNetworksArgs,
        SanityChecksForRelationNetworksResults,
    ]
):
    def _run(self):

        batch_size = self.args.batch_size
        device = torch.device(self.args.device)

        if self.args.dataset:
            dataset_dir = Path(self.args.dataset)
        else:
            dataset_dir = utils.clevr_xai_path()

        with open(utils.clevr_path() / "dic.pkl", "rb") as f:
            dic = pickle.load(f)

        n_words = len(dic["word_dic"]) + 1
        # n_answers = len(dic["answer_dic"])

        relnet = RelationNetworks(n_words)
        relnet = relnet.to(device)

        if self.args.model:
            model_dir = self.storage / self.args.model
            if not model_dir.exists():
                self.storage.download(model_dir)

            with open(model_dir / "results.pickle", "rb") as f:
                results = cast(train_clevr.TrainedModel, pickle.load(f))

            if self.args.checkpoint is None:
                ckpt = results.get_best_checkpoint()
                self.logger.info(
                    "No checkpoint given! Loading the best checkpoint..."
                )
            else:
                matching_ckpt = [
                    c
                    for c in results.checkpoints
                    if c.path == self.args.checkpoint
                ]
                if len(matching_ckpt) == 0:
                    raise ValueError(
                        f"No checkpoint found for {self.args.checkpoint}"
                    )
                ckpt = matching_ckpt[0]

            self.logger.info(f"Loading the weights of checkpoint: {ckpt}")

            with open(model_dir / "checkpoints" / ckpt.path, "rb") as f:
                relnet.load_state_dict(torch.load(f, map_location=device))

        print("-" * 80)
        print(relnet)
        print("-" * 80)

        # it is an unitialized model, does not matter which dataloader
        clevr_xai = data.get_clevr_xai_loader(
            dataset_dir,
            batch_size=batch_size,
            question_type=self.args.question_type,
            ground_truth=self.args.ground_truth,
        )

        dataset = itertools.islice(
            iter(clevr_xai), self.args.n_samples // batch_size
        )

        pbar = tqdm(dataset)

        saliency_0 = []
        saliency_1 = []
        saliency_rand_questions = []

        lrp_relnet = lrp.LRPViewOfRelationNetwork(relnet)
        lrp_relnet.to(device)

        get_random_answers = random_answer_same_category_fn(
            clevr_xai.dataset  # type: ignore
        )

        for i, (image, question, q_len, answer, dataset_index) in enumerate(
            pbar
        ):
            image, question, answer = (
                image.to(device),
                question.to(device),
                answer.to(device),
            )

            def get_lrp_saliency(
                target: torch.Tensor, randomize_questions: bool = False
            ) -> SaliencyResult:
                if randomize_questions:
                    q_perm = utils.randperm_without_fix_point(len(image))
                else:
                    q_perm = torch.arange(len(image))

                saliency = lrp_relnet.get_lrp_saliency(
                    image,
                    question,
                    q_len,
                    target,
                    q_perm,
                    normalize=False,
                )
                q_len_tensor = torch.tensor(q_len).cpu()
                image_idx = torch.tensor(dataset_index).cpu()
                question_index = image_idx[q_perm].contiguous().cpu()

                return SaliencyResult(
                    saliency=saliency.detach().cpu(),
                    images=image.detach().cpu(),
                    image_idx=image_idx,
                    question_len=q_len_tensor,
                    question=question.detach().cpu(),
                    question_index=question_index,
                    target=target.detach().cpu(),
                )

            saliency_0.append(get_lrp_saliency(answer))
            saliency_1.append(get_lrp_saliency(get_random_answers(answer)))
            saliency_rand_questions.append(
                get_lrp_saliency(answer, randomize_questions=True)
            )

        results = SanityChecksForRelationNetworksResults(
            SaliencyResult.cat(saliency_0),
            SaliencyResult.cat(saliency_1),
            SaliencyResult.cat(saliency_rand_questions),
        )

        print("-" * 80)
        print(results.statistics(lrp.normalize_saliency))
        print("-" * 80)

        return results
