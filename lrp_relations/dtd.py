import dataclasses
from typing import Callable, Union

import torch
from torch import nn
from torch.utils import hooks


class record_all_outputs:
    """A context manager that stores all outputs of all layers."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.outputs: dict[nn.Module, list[torch.Tensor]] = {}
        self.handles: list[hooks.RemovableHandle] = []

    def __enter__(
        self,
    ) -> dict[nn.Module, list[torch.Tensor]]:
        def hook(
            module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            if module not in self.outputs:
                self.outputs[module] = []

            self.outputs[module].append(output)

        self.module.apply(
            lambda module: self.handles.append(
                module.register_forward_hook(hook)
            )
        )
        return self.outputs

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


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

        # store parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = LinearReLU(input_size, hidden_size)
        self.layer2 = LinearReLU(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class NLayerMLP(nn.Module):
    def __init__(
        self, n_layers: int, input_size: int, hidden_size: int, output_size: int
    ):
        super().__init__()

        # store parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                layer = LinearReLU(input_size, hidden_size)
            elif i == n_layers - 1:
                layer = LinearReLU(hidden_size, output_size)
            else:
                layer = LinearReLU(hidden_size, hidden_size)
            setattr(self, f"layer{i + 1}", layer)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def root_point_linear(
    x: torch.Tensor,
    layer: LinearReLU,
    j: int,
    rule: str = "z+",
    gamma: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """Return the DTD root point.

    Args:
        x: Input tensor.
        layer: Layer to compute the root point.
        j: Index of which neuron to compute the root point.
        rule: Rule to compute the root point (supported: `z+`, `w2`, and
            `gamma`).
        gamma: Scaling factor for DTD gamma rule
    Returns:
        A root point `r` with the property `layer(r)[j] == 0`.
    """
    w = layer.linear.weight  # [out, in]
    b = layer.linear.bias  # [out]
    x  # [b, in]

    w_j = w[j, :].unsqueeze(0)
    b_j = b[j]
    indicator_w_j_pos = (w_j >= 0).float()  # [1, in]
    #  See 1.3 and 1.4 in DTD Appendix
    if rule == "z+":
        v = x * indicator_w_j_pos
    elif rule == "w2":
        v = w_j
    elif rule == "x":
        v = x
    elif rule == "zB":
        raise NotImplementedError()
    elif rule == "0":
        return 0 * x
    elif rule in ["gamma", "γ"]:
        # From: Layer-Wise Relevance Propagation: An Overview
        # https://www.doi.org/10.1007/978-3-030-28954-6_10
        # In section: 10.2.3 under Relation to LRP-0//γ

        # line eq: x - t * x * (1 + gamma * indicator_w_j_pos)
        # the intersection with the ReLU hinge is done below
        v = x * (1 + gamma * indicator_w_j_pos)
    else:
        raise ValueError()
    assert torch.allclose(layer.linear(x)[:, j].unsqueeze(1), x @ w_j.t() + b_j)

    #  This is equation (4) in DTD Appendix
    t = (x @ w_j.t() + b_j).sum(1) / (v @ w_j.t()).sum(1)
    # print("xw", (x @ w_j.t() + b_j).sum(1))
    # print("vw", (v @ w_j.t()).sum(1))
    # assert t.isnan().sum() == 0

    t[~t.isfinite()] = 0.0

    root_point = x - t.unsqueeze(1) * v
    return root_point


@dataclasses.dataclass
class LayerDTD:
    layer: LinearReLU
    layer_idx: int
    input: torch.Tensor
    output: torch.Tensor
    root: torch.Tensor


@dataclasses.dataclass
class PreciseDTD:
    net: NLayerMLP
    input: torch.Tensor
    explained_class: int
    rule: str = "z+"
    root_max_relevance: float = 1e-4
    max_search_tries: int = 10
    results: dict[LinearReLU, LayerDTD] = dataclasses.field(
        default_factory=dict
    )

    # Helper functions

    def is_first_layer(self, layer: LinearReLU) -> bool:
        return layer == self.net.layers[0]

    def is_final_layer(self, layer: LinearReLU) -> bool:
        return layer == self.net.layers[-1]

    def get_next_layer(self, layer: LinearReLU) -> LinearReLU:
        """Returns the next layer in the network."""
        return self.net.layers[self.net.layers.index(layer) + 1]

    def get_layer_index(self, layer: LinearReLU) -> int:
        return self.net.layers.index(layer)

    def get_prev_layer(self, layer: LinearReLU) -> LinearReLU:
        """Returns the previous layer in the network."""
        return self.net.layers[self.net.layers.index(layer) - 1]

    def get_input(self, layer: LinearReLU) -> torch.Tensor:
        """Returns the input of the layer."""
        if self.is_first_layer(layer):
            return self.input
        return self.outputs[self.get_prev_layer(layer)][0]

    def get_output(self, layer: LinearReLU) -> torch.Tensor:
        """Returns the output of the layer."""
        return self.outputs[layer][0]

    # Actual computation

    def compute_relevance_of_input(
        self,
        layer: LinearReLU,
        root: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the relevance of each neuron in a layer.

        Args:
            layer: Layer to compute the relevance.
            layer_activations: Activations of the layer.
            root: Root point of the layer.
        Returns:
            The relevance of each neuron in the layer.
        """

        if self.is_final_layer(layer):
            input = self.get_input(layer).clone()
            input.requires_grad_(True)
            out = layer.linear(input)
            R = out[:, self.explained_class]
            (dR_droot,) = torch.autograd.grad([R], [input], torch.ones_like(R))
        else:
            next_layer = self.get_next_layer(layer)
            root.requires_grad_(True)
            root_out = layer(root)
            R = self.compute_relevance_of_input(next_layer, root_out).sum(1)
            (dR_droot,) = torch.autograd.grad([R], [root], torch.ones_like(R))
        return dR_droot * (self.get_input(layer) - root)

    def _record_activations(self):
        with record_all_outputs(self.net) as self.outputs:
            self.net(self.input)

    def explain(self) -> torch.Tensor:
        self._record_activations()

        for layer in reversed(self.net.layers):
            root = self.find_root_point(layer)
            self.results[layer] = LayerDTD(
                layer,
                self.net.layers.index(layer),
                self.get_input(layer),
                self.get_output(layer),
                root,
            )
        first_layer = self.net.layers[0]
        return self.compute_relevance_of_input(
            first_layer, self.results[first_layer].root
        )

    def sample_search_space(
        self,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a search space for the given activation."""
        x = activation.new_empty(size=(1000, activation.shape[1]))
        x.uniform_(0, 1)
        return x

    def find_root_point(
        self,
        layer: LinearReLU,
    ) -> torch.Tensor:
        """Find the root point of a layer."""

        layer_input = self.get_input(layer)
        # layer_output = self.get_output(layer)
        if self.is_final_layer(layer):
            return root_point_linear(
                layer_input, layer, self.explained_class, self.rule
            )

        next_layer = self.get_next_layer(layer)

        for i in range(self.max_search_tries):
            coarse_search = self.sample_search_space(layer_input)
            coarse_hidden = layer(coarse_search)
            rel_coarse = self.compute_relevance_of_input(
                next_layer, coarse_hidden
            )
            mask = rel_coarse.sum(dim=1) <= self.root_max_relevance

            if mask.sum() > 1:
                candidates = coarse_search[mask.nonzero()]
                break

        if mask.sum() == 0:
            raise ValueError("Could not find suitable root point")

        steps = torch.linspace(-0.3, 1.05, 100).view(-1, 1)

        closest_relu_candidates = []
        distances_to_input = []
        for candidate in candidates:
            line_search = candidate + steps * (layer_input - candidate)
            line_hidden = layer(line_search)
            rel_line = self.compute_relevance_of_input(next_layer, line_hidden)
            root_idx = (
                (rel_line.sum(1) <= self.root_max_relevance).nonzero().max()
            )
            closest_relu_candidates.append(line_search[root_idx])
            distances_to_input.append(
                (layer_input - line_search[root_idx]).norm(p=2)
            )
        idx = torch.stack(distances_to_input).argmin()
        return closest_relu_candidates[idx]


def get_relevance_hidden(
    net: TwoLayerMLP,
    x: torch.Tensor,
    j: int = 0,
    rule: str = "z+",
    gamma: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    return get_relevance_hidden_and_root(net, x, j, rule, gamma)[0]


def get_relevance_hidden_and_root(
    net: TwoLayerMLP,
    x: torch.Tensor,
    j: int = 0,
    rule: str = "z+",
    gamma: Union[float, torch.Tensor] = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the relevance of each neuron in the hidden layer and the
    corresponding root point.

    This applies the given DTD rule to the final layer.

    Args:
        net: TwoLayerMLP network.
        x: Input tensor.
        j: Index of which neuron in the final layer to compute the relevance.
        rule: Rule to compute the root point
        gamma: Scaling factor for DTD gamma rule

    Returns:
        A tuple of two tensors: (the relevance of each neuron in the hidden
            layer, the corresponding root point).
    """
    with record_all_outputs(net) as outputs:
        net(x)

    # outputs
    hidden = outputs[net.layer1][0]
    hidden_root = root_point_linear(
        hidden, net.layer2, j, rule=rule, gamma=gamma
    )

    assert torch.isnan(hidden_root).sum() == 0

    output_root = net.layer2.linear(hidden_root)

    assert output_root.isnan().sum() == 0
    (rel_grad,) = torch.autograd.grad(
        output_root,
        hidden_root,
        grad_outputs=torch.ones_like(output_root),
        retain_graph=True,
    )
    assert rel_grad.isnan().sum() == 0
    rel_hidden = rel_grad * (hidden - hidden_root)
    return rel_hidden, hidden_root


def find_input_root_point(
    net: TwoLayerMLP,
    x: torch.Tensor,
    j: int,
    relevance_fn: Callable[[TwoLayerMLP, torch.Tensor], torch.Tensor],
    n_samples: int = 1_000,
    plot: bool = False,
) -> torch.Tensor:
    """Return the root point of the input layer.

    Args:
        net: TwoLayerMLP network.
        x: Input tensor.
        j: Index of which neuron in the hidden layer to compute the relevance.
        relevance_fn: Function to compute the relevance of each neuron in the
            hidden layer.
        n_samples: Number of samples to find the root point.

    Returns:
        The root point of the input layer.
    """
    assert x.size(0) == 1
    length = 3 * torch.randn(n_samples, 1) + 5
    x_search = x + torch.randn(n_samples, x.size(1)) * length

    rel_hidden_search = relevance_fn(net, x_search)

    idx = rel_hidden_search[:, j].abs().argmin()

    steps = torch.linspace(-0.3, 1.05, 100).view(-1, 1)

    smallest_x = x_search[idx].unsqueeze(0)
    x_line_search = smallest_x + steps * (x - smallest_x)

    rel_hidden_line_search = relevance_fn(net, x_line_search)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(rel_hidden_line_search.detach().cpu().numpy()[:, j])
        plt.show()
    root_idx = (rel_hidden_line_search[:, j] <= 1e-2).nonzero().max()
    return x_line_search[root_idx]


def almost_unique(x: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
    """Returns indices of unique elements in x, with tolerance atol."""
    sort_idx = torch.argsort(torch.norm(x, dim=1))
    x_sorted = x[sort_idx]

    dist = torch.pairwise_distance(x_sorted[:-1], x_sorted[1:])

    below_atol = (dist > atol).long()

    unique = dist.new_zeros(dist.shape[0] + 1, dtype=torch.long)
    unique[1:] = below_atol.cumsum(dim=0)

    unique_mask = x.new_zeros(size=(len(x),), dtype=torch.long)
    unique_mask[sort_idx] = unique
    return unique_mask
