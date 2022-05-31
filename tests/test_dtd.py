import pytest
import torch

from lrp_relations import dtd


def test_dtd_root():
    torch.manual_seed(0)
    net = dtd.TwoLayerMLP(input_size=3, hidden_size=5, output_size=1)

    idx = 0
    x = (0.25 * torch.randn(2, 3, requires_grad=True) + 1).clamp(min=0)

    rules: list[dtd.RULE] = ["0", "z+", "w2", dtd.GammaRule(1000)]
    for rule in rules:
        x_root = dtd.compute_root_for_single_neuron(
            x, net.layer1, idx, rule=rule
        )

        assert x_root is not None

        x_root.shape, x.shape
        print("x", x)
        print("x_root", x_root)
        print("out", net.layer1.linear(x)[:, idx].tolist())
        print("out root", net.layer1.linear(x_root)[:, idx].tolist())

        root_output = net.layer1.linear(x_root)[:, idx]
        assert torch.allclose(
            root_output, torch.zeros_like(root_output), atol=1e-6
        )


def test_dtd_input_root():
    torch.manual_seed(2)
    net = dtd.TwoLayerMLP(input_size=3, hidden_size=5, output_size=1)

    weight = net.layer2.linear.weight.data.clone()
    weight[:, :2] = 5 * weight.abs()[:, :2]
    net.layer2.linear.weight.data = weight
    net.layer2.linear.weight
    torch.manual_seed(3)
    x = (0.25 * torch.randn(1, 3, requires_grad=True) + 3).clamp(min=0)

    rules: list[dtd.RULE] = ["0", "z+", "w2", dtd.GammaRule(1000)]

    for rule in rules:
        print("-" * 80)
        print("Rule:", rule)

        def relevance_fn(net: dtd.TwoLayerMLP, x: torch.Tensor) -> torch.Tensor:
            return dtd.get_relevance_hidden(net, x, rule=rule)

        with dtd.record_all_outputs(net) as x_outs:
            logit_x = net(x)

        rel_hidden = relevance_fn(net, x)
        hidden_root = dtd.compute_root_for_single_neuron(
            x_outs[net.layer1][0], net.layer2, 0, rule=rule
        )
        print("hidden_root", hidden_root)
        print("x", x)
        x_root = dtd.find_input_root_point(
            net, x, 0, relevance_fn, n_samples=10_000
        )
        print("x_root", x_root)

        print("relevance hidden", rel_hidden)

        with dtd.record_all_outputs(net) as x_root_outs:
            logit_x_root = net(x_root)

        print("Logit x", logit_x.tolist())
        print("Logit x_root", logit_x_root.tolist())

        print("hidden with x", x_outs[net.layer1][0])
        print("hidden with x root", x_root_outs[net.layer1][0])


def test_almost_unique():
    atol = 1e-5
    x = torch.tensor(
        [[0, 0.0], [0, 2 * atol], [0, 2.2 * atol], [0, 0.1 * atol], [1, 1.0]]
    )

    unique_idx = dtd.almost_unique(x, atol=atol).tolist()
    assert unique_idx == [0, 1, 1, 0, 2]


@pytest.mark.skip(reason="TODO")
def test_dtd_precise():

    torch.manual_seed(0)

    x = torch.rand(1, 5)
    net = dtd.MLP(
        n_layers=3,
        input_size=5,
        hidden_size=10,
        output_size=2,
    )
    precise_dtd = dtd.PreciseDTD(
        net,
        explained_class=0,
        rule="z+",
        root_max_relevance=1e-3,
        n_random_samples=10,
    )
    with dtd.record_all_outputs(net) as outs:
        net(x)
    last_layer = net.layers[-1]
    input_last_layer = outs[net.layers[-2]][0]
    root_last_layer = precise_dtd.find_root_point(
        last_layer, input_last_layer, must_be_positive=True
    )
    assert (root_last_layer >= 0).all()
    assert root_last_layer.shape == (1, 10)

    random_root = torch.rand(50, 10)
    rel_random_root = precise_dtd.compute_relevance_of_input(
        last_layer, random_root
    )
    assert rel_random_root.shape == (50, 10)


@pytest.mark.skip(reason="TODO")
def test_dtd_precise_explain():

    torch.manual_seed(0)

    x = torch.rand(1, 5)
    net = dtd.MLP(
        n_layers=3,
        input_size=5,
        hidden_size=10,
        output_size=2,
    )
    precise_dtd = dtd.PreciseDTD(
        net,
        explained_class=0,
        rule="z+",
        root_max_relevance=1e-3,
        n_random_samples=10,
    )
    x = torch.rand(1, 5)
    ctx = precise_dtd.explain(x)
    assert ctx[net.layers[0]].relevance.shape == (1, 5)


def test_decompose_relevance_fns_full():
    torch.autograd.set_detect_anomaly(True)

    rule = dtd.rules.z_plus
    explained_output = 0

    torch.manual_seed(0)
    mlp = dtd.MLP(5, 10, 10, 2)
    mlp.init_weights()

    torch.manual_seed(0)
    for _ in range(100):
        x = torch.randn(1, mlp.input_size)
        if mlp(x)[:, explained_output] <= 0:
            continue
        break

    decomposed_fns = dtd.get_decompose_relevance_fns(
        mlp, explained_output=explained_output, rule=rule, decomposition="full"
    )

    with pytest.raises(ValueError):
        decomposed_fns[-1](x)


def test_decompose_relevance_fns_train_free():
    # torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float64)

    rule = dtd.rules.z_plus
    explained_output = 0

    torch.manual_seed(2)
    mlp = dtd.MLP(5, 10, 10, 2)
    mlp.init_weights()

    torch.manual_seed(4)
    for _ in range(1_000):
        x = torch.randn(1, mlp.input_size)
        if mlp(x)[:, explained_output] <= 0.25:
            continue
        break

    assert mlp(x)[:, explained_output] > 0.25

    decomposed_fns = dtd.get_decompose_relevance_fns(
        mlp,
        explained_output=explained_output,
        rule=rule,
        decomposition="train_free",
    )

    decomposition = decomposed_fns[-1](x)

    for rel in decomposition.collect_relevances():
        if isinstance(rel, dtd.TrainFreeRel):
            layer_idx = mlp.get_layer_index(rel.relevance_fn.to_input_of)
            print(
                layer_idx,
                rel.relevance.sum().item(),
                (rel.relevance < 0).sum().item(),
            )
        elif isinstance(rel, dtd.OutputRel):
            print("output", rel.relevance.sum())

    assert isinstance(decomposition, dtd.TrainFreeRel)
    assert decomposition.relevance.shape == (1, 10)
    assert decomposition.relevance.sum() > 0

    print(decomposition.relevance.sum())

    for rel in decomposition.collect_relevances():
        assert rel.relevance.sum() > 0

        assert torch.allclose(
            rel.relevance.sum(), decomposition.relevance.sum(), atol=1e-2
        )
