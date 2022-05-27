import torch
from lrp_relations import local_linear
from lrp_relations import dtd


def test_local_linear():

    torch.manual_seed(0)
    x = torch.rand(1, 5).repeat(40, 1)
    mlp = dtd.NLayerMLP(5, 5, 20, 1)

    print(mlp(x))
    assert (mlp(x) > 0).all()
    assert x.shape == (40, 5)
    chain, accepts = local_linear.sample(mlp, x, n_steps=200, n_warmup=0)
    assert chain.shape == (200, 40, 5)

    print(chain[-1].std(0))
    assert accepts.shape == (200,)
    assert accepts.mean() > 0.5
