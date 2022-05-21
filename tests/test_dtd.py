import torch

from lrp_relations import dtd


def test_dtd_root():
    torch.manual_seed(0)
    net = dtd.TwoLayerMLP(input_size=3, hidden_size=5, output_size=1)

    idx = 0
    x = (0.25 * torch.randn(2, 3, requires_grad=True) + 1).clamp(min=0)
    x_root = dtd.root_point(x, net.layer1, idx)

    x_root.shape, x.shape
    print("x", x)
    print("x_root", x_root)
    print("out", net.layer1.linear(x)[:, idx].tolist())
    print("out root", net.layer1.linear(x_root)[:, idx].tolist())

    root_output = net.layer1.linear(x_root)[:, idx]
    assert torch.allclose(root_output, torch.zeros_like(root_output))
