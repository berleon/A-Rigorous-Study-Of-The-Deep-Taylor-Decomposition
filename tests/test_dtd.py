import torch

from lrp_relations import dtd


def test_dtd_root():
    torch.manual_seed(0)
    net = dtd.TwoLayerMLP(input_size=3, hidden_size=5, output_size=1)

    idx = 0
    x = (0.25 * torch.randn(2, 3, requires_grad=True) + 1).clamp(min=0)

    for rule in ["0", "z+", "w2", "gamma"]:
        x_root = dtd.root_point_linear(x, net.layer1, idx, rule=rule)

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

    rules = ["0", "z+", "w2", "gamma"]

    for rule in rules:
        print("-" * 80)
        print("Rule:", rule)

        def relevance_fn(net: dtd.TwoLayerMLP, x: torch.Tensor) -> torch.Tensor:
            return dtd.get_relevance_hidden(net, x, rule=rule, gamma=1000)

        with dtd.record_all_outputs(net) as x_outs:
            logit_x = net(x)

        rel_hidden = relevance_fn(net, x)
        hidden_root = dtd.root_point_linear(
            x_outs[net.layer1][0], net.layer2, 0, rule=rule, gamma=1000
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


def test_dtd_precise():

    torch.manual_seed(0)

    x = torch.rand(1, 5)
    net = dtd.NLayerMLP(
        n_layers=3,
        input_size=5,
        hidden_size=10,
        output_size=2,
    )
    precise_dtd = dtd.PreciseDTD(
        net, x, explained_class=0, rule="z+", root_max_relevance=1e-3
    )
    precise_dtd._record_activations()
    last_layer = net.layers[-1]
    root_last_layer = precise_dtd.find_root_point(last_layer)
    assert (root_last_layer >= 0).all()
    assert root_last_layer.shape == (1, 10)

    random_root = torch.rand(50, 10)
    rel_random_root = precise_dtd.compute_relevance_of_input(
        last_layer, random_root
    )
    assert rel_random_root.shape == (50, 10)
    rel_input = precise_dtd.explain()
    assert isinstance(rel_input, torch.Tensor)
    assert rel_input.shape == (1, 5)
    assert rel_input.sum() >= 0.0
