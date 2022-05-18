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

import dataclasses
import itertools
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Callable

import captum.attr
import pandas as pd
import savethat
import torch
from captum.attr._utils import lrp_rules
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from relation_network.dataset import CLEVR, collate_data, transform
from relation_network.model import RelationNetworks

from lrp_relations import (  # noqa isort:skip
    enable_determistic,
)


class Sum(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


class Concat(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class ConcatRule(lrp_rules.EpsilonRule):
    def __init__(self, dim: int = 0) -> None:
        super().__init__(epsilon=1e-9)
        self.dim = dim

    def forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        def _create_backward_hook_input(
            input: torch.Tensor, start: int, end: int
        ) -> Callable[[torch.Tensor], None]:
            def _backward_hook_input(grad: torch.Tensor) -> None:
                rel = self.relevance_output[grad.device]
                idx = [slice(None, None, None) for _ in range(grad.dim())]
                idx[self.dim] = slice(start, end)
                return rel[idx]

            return _backward_hook_input

        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        inputs = lrp_rules._format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
        offset = 0
        for input in inputs:
            if not hasattr(input, "hook_registered"):
                next_offset = offset + input.size(self.dim)
                input_hook = _create_backward_hook_input(
                    input.data, offset, next_offset
                )
                offset = next_offset
                self._handle_input_hooks.append(input.register_hook(input_hook))
                input.hook_registered = True  # type: ignore
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)
        return outputs.clone()


class LRPViewOfRelationNetwork(nn.Module):
    def __init__(self, relnet: RelationNetworks):
        super().__init__()

        self.conv = deepcopy(relnet.conv)
        self.f = deepcopy(relnet.f)
        self.g = deepcopy(relnet.g)
        self.coords = deepcopy(relnet.coords)

        self.conv_hidden = relnet.conv_hidden
        self.lstm_hidden = relnet.lstm_hidden
        self.mlp_hidden = relnet.mlp_hidden
        self.n_concat = relnet.n_concat

        self.concat = Concat(dim=2)
        self.sum = Sum(dim=1)

    def forward(self, image, lstm_embed):
        conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        h_tile = lstm_embed.permute(1, 0, 2).expand(
            batch_size, n_pair * n_pair, self.lstm_hidden
        )

        conv = torch.cat(
            [conv, self.coords.expand(batch_size, 2, conv_h, conv_w)], 1
        )
        n_channel += 2
        conv_tr = conv.view(batch_size, n_channel, -1).permute(0, 2, 1)
        conv1 = conv_tr.unsqueeze(1).expand(
            batch_size, n_pair, n_pair, n_channel
        )
        conv2 = conv_tr.unsqueeze(2).expand(
            batch_size, n_pair, n_pair, n_channel
        )
        conv1 = conv1.contiguous().view(-1, n_pair * n_pair, n_channel)
        conv2 = conv2.contiguous().view(-1, n_pair * n_pair, n_channel)

        concat_vec = self.concat(conv1, conv2, h_tile).view(-1, self.n_concat)
        g = self.g(concat_vec)
        # return g.view(batch_size, -1)
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden)
        g = self.sum(g).squeeze()
        f = self.f(g)
        return f


def set_lrp_rules(lrp_relnet: nn.Module) -> None:
    for module in lrp_relnet.modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.Linear, Sum)):
            module.rule = lrp_rules.Alpha1_Beta0_Rule(set_bias_to_zero=False)
        elif isinstance(module, Concat):
            module.rule = ConcatRule(module.dim)
        else:
            module.rule = lrp_rules.IdentityRule()


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
        reverse_question = True
        n_worker = 9
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
        train_set = DataLoader(
            CLEVR(
                self.args.dataset,
                transform=transform,
                reverse_question=reverse_question,
                use_preprocessed=True,
            ),
            batch_size=batch_size,
            num_workers=n_worker,
            shuffle=True,
            collate_fn=collate_data,
        )

        dataset = itertools.islice(
            iter(train_set), self.args.n_samples // batch_size
        )

        pbar = tqdm(dataset)

        saliency_0 = []
        saliency_1 = []
        saliency_rand_questions = []

        lrp_relnet = LRPViewOfRelationNetwork(relnet)
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
                set_lrp_rules(lrp_relnet)
                lrp = captum.attr.LRP(lrp_relnet)

                question_embed = relnet.lstm_embed(question, q_len)

                if randomize_questions:
                    index = torch.randperm(question_embed.shape[1])
                    shape = question_embed.shape
                    question_embed = question_embed[:, index].contiguous()
                    print(question_embed.shape, shape)
                    assert question_embed.shape == shape

                saliency = lrp.attribute(
                    image,
                    target=target,
                    additional_forward_args=(question_embed,),
                    verbose=False,
                )

                output_lrp = lrp_relnet(image, question_embed)[:, target]

                # to make the heatmaps comparable we normalize the saliencies
                return saliency / output_lrp[:, None, None, None]

            saliency_0.append(get_normalized_lrp_saliency(0).cpu().detach())
            saliency_1.append(get_normalized_lrp_saliency(1).cpu().detach())
            saliency_rand_questions.append(
                get_normalized_lrp_saliency(0, randomize_questions=True)
                .cpu()
                .detach()
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
