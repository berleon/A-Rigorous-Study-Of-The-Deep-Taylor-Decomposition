from typing import cast

import captum.attr
import torch

from lrp_relations import dtd, local_linear, lrp


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


def test_decompose_relevance_fns_full():
    torch.autograd.set_detect_anomaly(True)

    rule = dtd.rules.z_plus
    explained_output = slice(0, 1)

    torch.manual_seed(0)
    mlp = dtd.MLP(3, 10, 10, 2)
    mlp.init_weights()

    x = mlp.get_input_with_output_greater(0.10, explained_output)

    root_finder = dtd.LinearDTDRootFinder(mlp, 0, rule)

    rel_fn_builder = dtd.FullBackwardFn.get_fn_builder(
        mlp,
        root_finder=root_finder,
        check_nans=True,
        stabilize_grad=dtd.StabilizeGradient(noise_std=1e-7, max_tries=10),
    )
    decomposed_fns = dtd.get_decompose_relevance_fns(
        mlp, explained_output, rel_fn_builder
    )

    out = decomposed_fns[-1](x)
    assert torch.isfinite(out.relevance).all()


def test_metropolis_hasting_root_finder():
    torch.manual_seed(2)
    mlp = dtd.MLP(3, 10, 10, 2)
    mlp.init_weights()
    explained_output = 0

    torch.manual_seed(4)
    for _ in range(100):
        x = torch.randn(1, mlp.input_size)
        if mlp(x)[:, explained_output] <= 0.25:
            continue
        break

    assert mlp(x)[:, explained_output] > 0.25

    root_finder = dtd.MetropolisHastingRootFinder(
        mlp,
        args=local_linear.MetropolisHastingArgs(
            n_steps=50,
            n_warmup=10,
        ),
    )

    rel_fn = dtd.NetworkOutputRelevanceFn(
        mlp, mlp.first_layer, explained_output
    )
    roots = root_finder.get_root_points_for_layer(
        mlp.first_layer, x, relevance_fn=rel_fn
    )
    assert len(roots) == 1
    assert roots[0].root.shape == (1, 10)

    rel_fn_builder = dtd.FullBackwardFn.get_fn_builder(
        mlp,
        root_finder=root_finder,
        stabilize_grad=None,
    )

    rel_fns = dtd.get_decompose_relevance_fns(
        mlp,
        explained_output,
        rel_fn_builder,
    )

    out = rel_fns[-2](x)

    assert root_finder.cache_hits > 0
    assert root_finder.cache_misses > 0

    assert torch.isfinite(out.relevance).all()


def test_sample_roots_interpolate():
    torch.manual_seed(1)
    mlp = dtd.MLP(3, 10, 10, 2)
    mlp.init_weights()
    explained_output = slice(0, 1)
    x = mlp.get_input_with_output_greater(
        0.5, explained_output, non_negative=True
    )

    mlp_output = mlp.slice(output=explained_output)

    root_finder = dtd.InterpolationRootFinder(
        mlp_output,
        args=local_linear.InterpolationArgs(
            batch_size=50,
            n_refinement_steps=10,
            n_batches=1,
            show_progress=True,
            enforce_non_negative=True,
        ),
    )

    network_output_fn = dtd.NetworkOutputRelevanceFn(
        mlp_output, mlp.first_layer, explained_output
    )

    roots = root_finder.get_root_points_for_layer(
        mlp.first_layer, x, relevance_fn=network_output_fn
    )

    assert len(roots) > 0

    rel_fn_builder = dtd.FullBackwardFn.get_fn_builder(
        mlp,
        root_finder=root_finder,
        stabilize_grad=None,
    )

    rel_fns = dtd.get_decompose_relevance_fns(
        mlp, explained_output, rel_fn_builder
    )
    rel_result = rel_fns[-1](x)
    assert torch.isfinite(rel_result.relevance).all()


def test_decompose_relevance_fns_train_free():
    # torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float64)

    rule = dtd.rules.z_plus
    explained_output = 0
    output_slice = slice(0, 1)

    torch.manual_seed(2)
    mlp = dtd.MLP(5, 10, 10, 2)
    mlp.init_weights()

    x = mlp.get_input_with_output_greater(0.25, output_slice, non_negative=True)

    rel_fn_builder = dtd.TrainFreeFn.get_fn_builder(
        mlp,
        root_finder=dtd.LinearDTDRootFinder(mlp, explained_output, rule),
    )
    decomposed_fns = dtd.get_decompose_relevance_fns(
        mlp,
        explained_output,
        rel_fn_builder,
    )

    decomposition = decomposed_fns[-1](x)

    for rel in decomposition.collect_relevances():
        if isinstance(rel, dtd.TrainFreeRel):
            layer_idx = mlp.get_layer_index(
                rel.computed_with_fn.get_lower_rel_layer()
            )
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


def test_dtd_relevances_sumup():
    explained_output: slice = slice(0, 1)
    rule = dtd.rules.z_plus

    torch.manual_seed(2)
    mlp = dtd.MLP(1, 10, 10, 2)
    mlp.init_weights()

    torch.manual_seed(1)

    x = mlp.get_input_with_output_greater(
        0.25, explained_output, non_negative=True
    )

    root_finder = dtd.LinearDTDRootFinder(
        mlp,
        explained_output.start,
        rule,
    )

    input_layer = mlp.first_layer

    output_fn = dtd.OutputRelFn(mlp, explained_output)
    output_rel = output_fn(input_layer(x))

    const_rel = dtd.ConstantRelFn(
        mlp,
        input_layer,
        input_layer,
        output_fn.get_upper_rel_layer(),
        output_rel.relevance,
    )

    roots = root_finder.get_root_points_for_layer(mlp.first_layer, x, const_rel)

    for root in roots:
        assert root.root.shape == (1, 10)

    sum_relevance = sum(root.relevance for root in roots)
    assert sum_relevance == output_rel.relevance.sum()


def test_dtd_train_free_matches_captum():
    explained_output: slice = slice(0, 1)
    rule = dtd.rules.z_plus

    torch.manual_seed(2)
    mlp = dtd.MLP(4, 10, 10, 2)
    mlp.init_weights()

    torch.manual_seed(1)

    x = mlp.get_input_with_output_greater(
        0.25, explained_output, non_negative=True
    )

    root_finder = dtd.LinearDTDRootFinder(
        mlp,
        explained_output.start,
        rule,
    )

    rel_fn_builder = dtd.TrainFreeFn.get_fn_builder(
        mlp,
        root_finder=root_finder,
        check_consistent=True,
    )

    rel_fns = dtd.get_decompose_relevance_fns(
        mlp, explained_output, rel_fn_builder
    )

    mlp_output = mlp.slice(output=explained_output)
    logit = mlp_output(x)

    rel_result = cast(dtd.TrainFreeRel, rel_fns[-1](x))

    rel_result.relevance
    lrp.set_lrp_rules(mlp, set_bias_to_zero=True)
    lrp_attr = captum.attr.LRP(mlp)
    saliency = lrp_attr.attribute(x, target=explained_output.start)

    assert torch.allclose(saliency.sum(), logit)

    assert torch.allclose(
        rel_result.relevance,
        saliency,
        atol=1e-5,
    )
