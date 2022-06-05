import torch

from lrp_relations import dtd, local_linear


def test_local_linear():

    torch.manual_seed(1)
    mlp = dtd.MLP(5, 5, 20, 1)
    mlp.init_weights()
    for _ in range(100):
        x = torch.rand(1, 5).repeat(40, 1)
        if (mlp(x) > 0).all():
            break

    print(mlp(x))
    assert (mlp(x) > 0).all()
    assert x.shape == (40, 5)
    args = local_linear.MetropolisHastingArgs(n_steps=200, n_warmup=0)
    sampling = local_linear.sample_metropolis_hasting(mlp, x, args)
    chain = sampling.chain
    assert chain.shape == (200, 40, 5)
    print(chain[-1].std(0))

    accepts = sampling.accept_ratio
    assert accepts.shape == (200,)
    assert accepts.mean() > 0.5


def test_sample_interpolation():

    torch.manual_seed(1)
    mlp = dtd.MLP(5, 5, 20, 1)
    mlp.init_weights()
    for _ in range(100):
        x = torch.rand(1, 5)
        if (mlp(x) > 0).all():
            break

    assert (mlp(x) > 0).all()

    x.requires_grad_(True)

    out = mlp(x)
    (sample_grad,) = torch.autograd.grad(
        out, x, grad_outputs=torch.ones_like(out)
    )
    sample_grad = sample_grad[:1]

    assert (mlp(x) > 0).all()

    result = local_linear.sample_interpolation(mlp, x)

    assert (result.edge_tolerance < 1e-3).all()

    for samples in [result.all_valid_points, result.edge_points]:
        assert samples.ndim == 2
        assert samples.shape[0] > 1
        samples.requires_grad_(True)

        out = mlp(samples)
        (grad,) = torch.autograd.grad(
            out, samples, grad_outputs=torch.ones_like(out)
        )
        assert torch.allclose(sample_grad.repeat(len(grad), 1), grad, atol=1e-5)
