import torch
from torch import nn


class LinearReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

        def clamp_biases(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                module.bias.data.clamp_(max=0)

        self.apply(clamp_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layer1 = LinearReLU(input_size, hidden_size)
        self.layer2 = LinearReLU(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


def root_point(
    x: torch.Tensor, layer: LinearReLU, j: int, rule: str = "z+"
) -> torch.Tensor:
    """Return the DTD root point.

    Args:
        x: Input tensor.
        layer: Layer to compute the root point.
        j: Index of which neuron to compute the root point.
        rule: Rule to compute the root point (supported: `z+` and `w2`).

    Returns:
        A root point `r` with the property `layer(r)[j] == 0`.
    """
    w = layer.linear.weight  # [out, in]
    b = layer.linear.bias  # [out]
    x  # [b, in]

    #  See 1.3 and 1.4 in DTD Appendix
    if rule == "z+":
        v = x * (w[j, :] >= 0).unsqueeze(0)
    elif rule == "w2":
        v = w[j, :].unsqueeze(0)
    elif rule == "zB":
        raise NotImplementedError()
    else:
        raise ValueError()

    w_j = w[j, :].unsqueeze(0)
    b_j = b[j]
    assert torch.allclose(layer.linear(x)[:, j].unsqueeze(1), x @ w_j.t() + b_j)

    #  This is equation (4) in DTD Appendix
    t = (x @ w_j.t() + b_j).sum(1) / (v @ w_j.t()).sum(1)
    return x - t.unsqueeze(1) * v
