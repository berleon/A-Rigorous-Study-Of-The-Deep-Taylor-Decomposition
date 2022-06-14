"""Implementation of the LRP_a1_b0 rules for the Relation Network."""

from copy import deepcopy
from typing import Callable, Optional, Union, cast

import captum.attr
import numpy as np
import torch
from captum.attr._utils import lrp_rules
from torch import nn

from lrp_relations import utils
from relation_network.model import RelationNetworks


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
                rel = self.relevance_output[grad.device]  # type: ignore
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

        # do not include in the modules
        self._relnet = [relnet]
        self.conv = deepcopy(relnet.conv)
        self.f = deepcopy(relnet.f)
        self.g = deepcopy(relnet.g)
        self.coords = cast(torch.Tensor, deepcopy(relnet.coords))

        self.conv_hidden = relnet.conv_hidden
        self.lstm_hidden = relnet.lstm_hidden
        self.mlp_hidden = relnet.mlp_hidden
        self.n_concat = relnet.n_concat

        self.concat = Concat(dim=2)
        self.sum = Sum(dim=1)

    @property
    def original_relnet(self) -> RelationNetworks:
        return self._relnet[0]

    def forward(self, image, lstm_embed):
        conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        h_tile = lstm_embed.permute(1, 0, 2).expand(
            batch_size, n_pair * n_pair, self.lstm_hidden
        )

        coords = self.coords.expand(batch_size, 2, conv_h, conv_w)
        conv = torch.cat([conv, coords], 1)
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

    def get_lrp_saliency(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        q_len: int,
        target: Union[int, torch.Tensor],
        question_permutation: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.get_lrp_saliency_and_logits(
            image,
            question,
            q_len,
            target,
            question_permutation,
            normalize,
        )[0]

    def get_lrp_saliency_and_logits(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        q_len: int,
        target: Union[int, torch.Tensor],
        question_permutation: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        set_lrp_rules(self)
        lrp_attr = captum.attr.LRP(self)

        if isinstance(target, int):
            target = image.new_full((image.size(0),), target, dtype=torch.long)

        question_embed = self.original_relnet.lstm_embed(question, q_len)

        if question_permutation is not None:
            shape = question_embed.shape
            question_embed = question_embed[
                :, question_permutation
            ].contiguous()
            assert question_embed.shape == shape

        saliency = lrp_attr.attribute(
            image,
            target=target,
            additional_forward_args=(question_embed,),
            verbose=False,
        )

        logits = self(image, question_embed)

        if normalize:
            explained_logit = logits[torch.arange(len(logits)), target]
            # to make the heatmaps comparable we normalize the saliencies
            saliency = saliency / explained_logit[:, None, None, None]

        return saliency, logits


def set_lrp_rules(lrp_relnet: nn.Module, set_bias_to_zero: bool = True) -> None:
    for module in lrp_relnet.modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.Linear, Sum)):
            module.rule = lrp_rules.Alpha1_Beta0_Rule(
                set_bias_to_zero=set_bias_to_zero
            )
        elif isinstance(module, Concat):
            module.rule = ConcatRule(module.dim)
        else:
            module.rule = lrp_rules.IdentityRule()


def normalize_saliency(
    saliency: torch.Tensor,
    clip_percentile_max: Optional[float] = 99.5,
    clip_percentile_min: Optional[float] = 0.5,
    retain_zero: bool = False,
    abs: bool = True,
) -> torch.Tensor:
    assert not retain_zero
    if abs:
        saliency = saliency.abs()
    saliency_np = utils.to_np(saliency)
    vmin, vmax = np.percentile(
        saliency_np, [clip_percentile_min or 0, clip_percentile_max or 100]
    )
    saliency = saliency.clamp(vmin, vmax)
    return (saliency - vmin) / (vmax - vmin)
