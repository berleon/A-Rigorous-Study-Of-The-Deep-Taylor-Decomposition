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

from lrp_relations import enable_deterministic  # noqa isort:skip

import dataclasses
import itertools
import pickle
from pathlib import Path

import pandas as pd
import savethat
import torch
from tqdm import tqdm

from lrp_relations import data, lrp
from relation_network.model import RelationNetworks


@dataclasses.dataclass(frozen=True)
class SanityChecksForRelationNetworksArgs(savethat.Args):
    dataset: str = "../data/clevr/CLEVR_v1.0/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples: int = 1000


@dataclasses.dataclass(frozen=True)
class SanityChecksForRelationNetworksResults:
    saliency_0: torch.Tensor
    saliency_1: torch.Tensor
    saliency_0_rand_questions: torch.Tensor

    def statistics(self) -> pd.DataFrame:
        diff_01 = (self.saliency_0 - self.saliency_1).abs()
        diff_question = (self.saliency_0 - self.saliency_0_rand_questions).abs()

        items = [
            self.saliency_0,
            self.saliency_1,
            self.saliency_0_rand_questions,
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


class SanityChecksForRelationNetworks(
    savethat.Node[SanityChecksForRelationNetworksArgs, bool]
):
    def _run(self):

        batch_size = 30
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset_dir = Path(self.args.dataset)
        with open(dataset_dir / "dic.pkl", "rb") as f:
            dic = pickle.load(f)

        n_words = len(dic["word_dic"]) + 1
        # n_answers = len(dic["answer_dic"])

        relnet = RelationNetworks(n_words)
        relnet = relnet.to(device)

        print("-" * 80)
        print(relnet)
        print("-" * 80)

        # it is an unitialized model, does not matter which dataloader
        train_set = data.get_clevr_dataloader(
            dataset_dir, "train", batch_size=batch_size
        )

        dataset = itertools.islice(
            iter(train_set), self.args.n_samples // batch_size
        )

        pbar = tqdm(dataset)

        saliency_0 = []
        saliency_1 = []
        saliency_rand_questions = []

        lrp_relnet = lrp.LRPViewOfRelationNetwork(relnet)
        lrp_relnet.to(device)

        for i, (image, question, q_len, answer, _) in enumerate(pbar):
            image, question, q_len, answer = (
                image.to(device),
                question.to(device),
                torch.tensor(q_len),
                answer.to(device),
            )

            def get_normalized_lrp_saliency(
                target: int, randomize_questions: bool = False
            ) -> torch.Tensor:
                saliency = lrp_relnet.get_lrp_saliency(
                    image, question, q_len, target, randomize_questions
                )
                return saliency.cpu().detach()

            saliency_0.append(get_normalized_lrp_saliency(0))
            saliency_1.append(get_normalized_lrp_saliency(1))
            saliency_rand_questions.append(
                get_normalized_lrp_saliency(0, randomize_questions=True)
            )

        results = SanityChecksForRelationNetworksResults(
            torch.cat(saliency_0, dim=0),
            torch.cat(saliency_1, dim=0),
            torch.cat(saliency_rand_questions, dim=0),
        )

        print("-" * 80)
        print(results.statistics())
        print("-" * 80)

        return results
