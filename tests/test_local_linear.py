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
    sampling = local_linear.sample(mlp, x, n_steps=200, n_warmup=0)
    chain = sampling.chain
    assert chain.shape == (200, 40, 5)
    print(chain[-1].std(0))

    accepts = sampling.accept_ratio
    assert accepts.shape == (200,)
    assert accepts.mean() > 0.5
