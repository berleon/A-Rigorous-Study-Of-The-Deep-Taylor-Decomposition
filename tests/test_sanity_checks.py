#!/usr/bin/env python
"""Tests for `sanity_checks_for_relation_networks` package."""

from lrp_relations import enable_deterministic, utils  # noqa isort:skip

import pickle
from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from lrp_relations import lrp
from relation_network.dataset import CLEVR, collate_data, transform
from relation_network.model import RelationNetworks


@pytest.fixture
def dataset_dir() -> Path:
    return utils.clevr_path()


try:
    clevr_available = utils.clevr_path().exists()
except ValueError:
    clevr_available = False


@pytest.mark.skipif(not clevr_available, reason="CLEVR DATASET not available")
def test_lrp_relnet_matches_output(dataset_dir: Path) -> None:

    with open(dataset_dir / "dic.pkl", "rb") as f:
        dic = pickle.load(f)

    batch_size = 30
    reverse_question = True
    n_worker = 9
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_words = len(dic["word_dic"]) + 1
    relnet = RelationNetworks(n_words)
    relnet.to(device)
    lrp_relnet = lrp.LRPViewOfRelationNetwork(relnet)
    lrp_relnet.to(device)

    train_set: DataLoader = DataLoader(
        CLEVR(
            dataset_dir,
            transform=transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
        ),
        batch_size=batch_size,
        num_workers=n_worker,
        shuffle=True,
        collate_fn=collate_data,
    )

    dataset = iter(train_set)
    image, question, q_len, answer, _ = next(dataset)

    image, question, q_len, answer = (
        image.to(device),
        question.to(device),
        torch.tensor(q_len),
        answer.to(device),
    )

    def get_store_hook(
        output_list: list[torch.Tensor],
    ) -> Callable[[nn.Module, tuple[torch.Tensor], torch.Tensor], None]:
        def store_output(
            module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            output_list.append(output)

        return store_output

    conv_outputs: list[torch.Tensor] = []
    relnet.conv.register_forward_hook(get_store_hook(conv_outputs))
    lrp_relnet.conv.register_forward_hook(get_store_hook(conv_outputs))

    g_outputs: list[torch.Tensor] = []
    relnet.g.register_forward_hook(get_store_hook(g_outputs))
    lrp_relnet.g.register_forward_hook(get_store_hook(g_outputs))

    lstm_outputs: list[torch.Tensor] = []
    relnet.lstm.register_forward_hook(get_store_hook(lstm_outputs))

    f_inputs = []

    def store_f_inputs(
        module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        f_inputs.append(inputs[0])

    relnet.f.register_forward_hook(store_f_inputs)
    lrp_relnet.f.register_forward_hook(store_f_inputs)

    target = 0
    question_embed = relnet.lstm_embed(question, q_len)
    output_lrp = lrp_relnet(image.clone(), question_embed.clone())[:, target]
    output_relnet = relnet(image, question, q_len)[:, target]

    assert len(conv_outputs) == 2
    assert torch.allclose(conv_outputs[0], conv_outputs[1])

    # get hidden state of lstm
    lstm_hidden = [o[1][0] for o in lstm_outputs]
    assert torch.allclose(lstm_hidden[0], lstm_hidden[1])

    assert len(g_outputs) == 2
    assert torch.allclose(g_outputs[0], g_outputs[1])

    assert len(f_inputs) == 2
    assert torch.allclose(f_inputs[0], f_inputs[1])

    assert torch.allclose(output_lrp, output_relnet)
