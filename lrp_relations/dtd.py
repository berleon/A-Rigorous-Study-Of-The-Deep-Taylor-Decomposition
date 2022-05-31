from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.utils import hooks
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Model


class LinearReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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


class MLP(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        first: Optional[LinearReLU] = None,
        last: Optional[LinearReLU] = None,
    ) -> torch.Tensor:
        if first is None:
            first = self.layers[0]
        if last is None:
            last = self.layers[-1]

        first_seen = False

        for layer in self.layers:
            if layer == first:
                first_seen = True

            if first_seen:
                x = layer(x)
            if layer == last:
                break
        return x

    # helper functions

    def get_all_outputs(
        self,
        x: torch.Tensor,
        first: Optional[LinearReLU] = None,
        last: Optional[LinearReLU] = None,
    ) -> dict[nn.Module, torch.Tensor]:
        """Returns a dictionary of all outputs of the network."""
        with record_all_outputs(self) as outputs:
            self(x, first, last)
        return {mod: out[0] for mod, out in outputs.items()}

    def is_first_layer(self, layer: LinearReLU) -> bool:
        return layer == self.layers[0]

    def is_final_layer(self, layer: LinearReLU) -> bool:
        return layer == self.layers[-1]

    def get_next_layer(self, layer: LinearReLU) -> LinearReLU:
        """Returns the next layer in the network."""
        if self.is_final_layer(layer):
            raise ValueError("Cannot get next layer of last layer.")
        return self.layers[self.layers.index(layer) + 1]

    def get_layer_index(self, layer: LinearReLU) -> int:
        return self.layers.index(layer)

    def get_prev_layer(self, layer: LinearReLU) -> LinearReLU:
        """Returns the previous layer in the network."""
        if self.is_first_layer(layer):
            raise ValueError("Cannot get previous layer of first layer.")
        return self.layers[self.layers.index(layer) - 1]


def grad_mask_for_logit(output: torch.Tensor, index: int) -> torch.Tensor:
    mask = output.new_zeros(output.shape, dtype=torch.float)
    mask[:, index] = 1.0
    return mask


# -----------------------------------------------------------------------------
# Linear Root points


class Rule:
    def __init__(self, name_or_rule: RULE):
        if isinstance(name_or_rule, str):
            self.name = name_or_rule
        else:
            assert isinstance(name_or_rule, Rule)
            self.name = name_or_rule.name

        self.is_layer_rule = self.name in ["pinv"]


def rule_name(rule: RULE) -> str:
    if isinstance(rule, str):
        return rule
    elif isinstance(rule, Rule):
        return rule.name
    else:
        raise ValueError(f"Unknown rule {rule}")


RULE = Union[Rule, str]


class GammaRule(Rule):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        super().__init__("gamma")


class rules:
    pinv = Rule("pinv")
    z_plus = Rule("z+")
    w2 = Rule("w2")
    x = Rule("x")
    zB = Rule("zB")

    @staticmethod
    def gamma(gamma: float) -> GammaRule:
        return GammaRule(gamma=gamma)


def is_layer_rule(rule: RULE) -> bool:
    """Returns whether the rule is a layer rule."""
    return rule == "pinv"


def compute_root_for_layer(
    x: torch.Tensor,
    layer: LinearReLU,
    rule: RULE = "pinv",
) -> torch.Tensor:
    w = layer.linear.weight  # [out, in]
    b = layer.linear.bias  # [out]

    if rule_name(rule) == "pinv":
        return (torch.linalg.pinv(w) @ b).unsqueeze(0)
    else:
        raise ValueError(f"Unknown rule {rule}")


def compute_root_for_single_neuron(
    x: torch.Tensor,
    layer: LinearReLU,
    j: int,
    rule: RULE = "z+",
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

    name = rule_name(rule)
    if name == "z+":
        v = x * indicator_w_j_pos
    elif name == "w2":
        v = w_j
    elif name == "x":
        v = x
    elif name == "zB":
        raise NotImplementedError()
    elif name == "0":
        return 0 * x
    elif name in ["gamma", "γ"]:
        # From: Layer-Wise Relevance Propagation: An Overview
        # https://www.doi.org/10.1007/978-3-030-28954-6_10
        # In section: 10.2.3 under Relation to LRP-0//γ

        # line eq: x - t * x * (1 + gamma * indicator_w_j_pos)
        # the intersection with the ReLU hinge is done below
        v = x * (1 + gamma * indicator_w_j_pos)
    else:
        raise ValueError()
    assert torch.allclose(
        layer.linear(x)[:, j].unsqueeze(1), x @ w_j.t() + b_j, atol=1e-6
    )

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
class RootConstraint:
    positive: bool  # root point must be positive
    gradient_close: bool  # root point must have the same gradient as input
    gradient_atol: float = 1e-6

    def is_fulfilled(
        self,
        root: torch.Tensor,
        input_grad: Optional[torch.Tensor] = None,
        root_grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = root.new_ones((len(root),), dtype=torch.bool)
        if self.positive:
            mask[(root < 0).any(1)] = False
        if self.gradient_close:
            if input_grad is None or root_grad is None:
                mask[:] = False
                return mask
            grad_close = torch.isclose(
                root.grad,
                input_grad,
                atol=self.gradient_atol,
            )
            mask[~grad_close] = False
        return mask


class RootFinding:
    constraint: RootConstraint


@dataclasses.dataclass
class RandomSampling(RootFinding):
    constraint: RootConstraint
    n_samples: int
    layer: LinearReLU
    root_max_relevance: float = 1e-4
    max_search_tries: int = 10


class PrecomputedRoot(RootFinding):
    def __init__(self, root: torch.Tensor):
        self.root = root

    def find(self, input: torch.Tensor) -> torch.Tensor:
        return self.root


@dataclasses.dataclass
class LayerRelContext:
    layer: LinearReLU
    # input
    input: torch.Tensor
    logit: torch.Tensor
    grad_input: torch.Tensor
    # root
    root: torch.Tensor
    root_logit: torch.Tensor
    grad_root: torch.Tensor
    # relevance
    relevance: torch.Tensor
    #
    # root_finding: RootFinding


@dataclasses.dataclass
class RelevanceContext:
    layers: list[LayerRelContext]


@dataclasses.dataclass
class LayerOptions:
    root_must_be_positive: bool = True


@dataclasses.dataclass
class PreciseDTD:
    net: MLP
    explained_class: int
    rule: str = "z+"
    root_max_relevance: float = 1e-4
    max_search_tries: int = 10
    n_random_samples: int = 50
    options: dict[LinearReLU, LayerOptions] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.options:
            self.options = {layer: LayerOptions() for layer in self.net.layers}

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

    # Actual computation

    def compute_relevance_of_input(
        self,
        layer: LinearReLU,
        input: torch.Tensor,
        context: Optional[dict[LinearReLU, LayerRelContext]] = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Compute the relevance of each neuron in a layer.

        Args:
            layer: Layer to compute the relevance.
            input: The input to the layer.
        Returns:
            The relevance of each neuron in the layer.
        """
        root = self.find_root_point(
            layer,
            input,
            self.options[layer].root_must_be_positive,
            show_progress,
        )
        root.requires_grad_(True)

        if self.is_final_layer(layer):
            out = layer.linear(root)
            R = out[:, self.explained_class]
            (dR_droot,) = torch.autograd.grad([R], [root], torch.ones_like(R))
        else:
            next_layer = self.get_next_layer(layer)
            root.requires_grad_(True)
            root_out = layer(root)
            R = self.compute_relevance_of_input(next_layer, root_out).sum(1)
            (dR_droot,) = torch.autograd.grad([R], [root], torch.ones_like(R))

        relevance = dR_droot * (input - root)

        if context is not None:
            input.requires_grad_(True)
            logit = self.net(input, first=layer)
            grad_mask = grad_mask_for_logit(logit, self.explained_class)
            (grad_input,) = torch.autograd.grad([logit], [input], grad_mask)

            root_logit = self.net(root, first=layer)
            grad_mask = grad_mask_for_logit(root_logit, self.explained_class)
            (grad_root,) = torch.autograd.grad([root_logit], [root], grad_mask)

            context[layer] = LayerRelContext(
                layer=layer,
                input=input,
                logit=logit,
                grad_input=grad_input,
                root=root,
                root_logit=root_logit,
                grad_root=grad_root,
                relevance=relevance,
            )
        return relevance

    def explain(
        self, input: torch.Tensor, show_progress: bool = False
    ) -> dict[LinearReLU, LayerRelContext]:
        first_layer = self.net.layers[0]
        context: dict[LinearReLU, LayerRelContext] = {}

        self.compute_relevance_of_input(
            first_layer, input, context, show_progress
        )
        return context

    def sample_search_space(
        self,
        activation: torch.Tensor,
        must_be_positive: bool = False,
    ) -> torch.Tensor:
        """Sample a search space for the given activation."""
        x = activation.new_empty(
            size=(self.n_random_samples, activation.shape[1])
        )

        x.uniform_(0, 1)
        if must_be_positive:
            x = x.clamp(min=0)
        return x

    def find_root_point(
        self,
        layer: LinearReLU,
        input: torch.Tensor,
        must_be_positive: bool = False,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Find the root point of a layer.

        Args:
            layer: The layer to find the root point of.
            input: The input to the layer.

        Returns:
            The root point of the layer.
        """
        if input.dim == 1:
            input.unsqueeze_(0)

        def get_relevance(x: torch.Tensor) -> torch.Tensor:
            if next_layer is None:  # is_final_layer == True
                # this is the final layer
                # just use the explained class as the relevance
                R = x[:, self.explained_class].unsqueeze(1)
                return R
            else:
                return self.compute_relevance_of_input(next_layer, x)

        is_final_layer = self.is_final_layer(layer)
        next_layer = self.get_next_layer(layer) if not is_final_layer else None

        roots = []
        for sample_idx, single_input in enumerate(input):
            single_input = single_input.unsqueeze(0)

            pbar = tqdm(
                list(range(self.max_search_tries)),
                disable=not show_progress,
                desc=f"[{sample_idx}/{len(input)}] Random sampling",
            )
            for i in pbar:
                # TODO: restrict the search space
                coarse_search = self.sample_search_space(
                    single_input, must_be_positive
                )

                coarse_hidden = layer(coarse_search)
                rel_coarse = get_relevance(coarse_hidden)

                mask = rel_coarse.sum(dim=1) <= self.root_max_relevance

                if mask.sum() > 1:
                    candidates = coarse_search[mask.nonzero()]
                    break
            print("stopped at", i)
            if mask.sum() == 0:
                raise ValueError("Could not find suitable root point")

            closest_relu_candidates = []
            distances_to_input = []

            pbar_candidates = tqdm(
                candidates,
                disable=not show_progress,
                desc=f"[{sample_idx}/{len(input)}] Fine-tune sampling",
            )
            for candidate in pbar_candidates:
                diff_to_input = single_input - candidate

                def get_center_factor():
                    return lower + (upper - lower) / 2

                def get_center():
                    mid = get_center_factor()
                    return candidate + mid * diff_to_input

                lower = -0.3
                upper = 1.05
                while True:
                    center = get_center()
                    rel = get_relevance(center)
                    if rel.sum() > self.root_max_relevance:
                        upper = get_center_factor()
                    else:
                        lower = get_center_factor()

                    diff = (rel.sum() - self.root_max_relevance).abs()
                    print(diff)
                    if diff < 1e-3:
                        break

                closest_relu_candidates.append(center)
                distances_to_input.append((input - center).norm(p=2))
                pbar_candidates.refresh()
            idx = torch.stack(distances_to_input).argmin()
            roots.append(closest_relu_candidates[idx])
        return torch.stack(roots)


@dataclasses.dataclass
class RecursiveRoot:
    root: torch.Tensor
    input: torch.Tensor
    layer: LinearReLU
    outputs_of_root: dict[nn.Module, torch.Tensor]
    outputs_of_input: dict[nn.Module, torch.Tensor]
    rule: Rule
    explained_neuron: Optional[int]
    explained_class: int
    upper_layers: list[RecursiveRoot]

    def select(self, layer: LinearReLU) -> "RecursiveRoot":
        """Select the root point of the given layer."""
        if layer is self.layer:
            return self
        else:
            for upper_layer in self.upper_layers:
                return upper_layer.select(layer)
        raise KeyError(f"Layer {layer} not found")


@dataclasses.dataclass
class RecursiveRoots:
    """Find the roots of a neural network."""

    mlp: MLP
    explained_class: int
    rule: Rule
    gamma: float = 0.1

    def get_root_points(
        self,
        layer: LinearReLU,
        input: torch.Tensor,
        end_at: LinearReLU,
    ) -> list[RecursiveRoot]:
        rule = getattr(layer, "rule", None) or self.rule
        gamma = getattr(layer, "gamma", None) or self.gamma

        if is_layer_rule(rule):
            roots = [compute_root_for_layer(input, layer, rule)]

        else:
            explained_neurons = list(range(layer.out_features))
            if self.mlp.is_final_layer(layer):
                explained_neurons = [self.explained_class]

            roots = [
                compute_root_for_single_neuron(input, layer, j, rule, gamma)
                for j in explained_neurons
            ]

        rr = []
        for j, root in enumerate(roots):
            if layer != end_at:
                upper_layers = self.get_root_points(
                    self.mlp.get_next_layer(layer), root, end_at
                )
            else:
                upper_layers = []
            outputs_for_root = self.mlp.get_all_outputs(root, first=layer)
            outputs_for_input = self.mlp.get_all_outputs(input, first=layer)
            rr.append(
                RecursiveRoot(
                    root=root,
                    outputs_of_root=outputs_for_root,
                    input=input,
                    outputs_of_input=outputs_for_input,
                    rule=Rule(rule),
                    layer=layer,
                    explained_neuron=j if is_layer_rule(rule) else None,
                    explained_class=self.explained_class,
                    upper_layers=upper_layers,
                )
            )
        return rr

    def run(
        self,
        input: torch.Tensor,
        start_at: Optional[LinearReLU] = None,
        end_at: Optional[LinearReLU] = None,
    ) -> list[RecursiveRoot]:
        """Run the recursive roots algorithm.

        Args:
            input: The input to the neural network.
            first_root_layer: The first layer to use as root.
        """
        if start_at is None:
            start_at = self.mlp.layers[0]
        if self.mlp.is_first_layer(start_at):
            hidden = input
        else:
            hidden = self.mlp(input, last=self.mlp.get_prev_layer(start_at))

        if end_at is None:
            end_at = self.mlp.layers[-1]

        return self.get_root_points(start_at, hidden, end_at)


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
    hidden_root = compute_root_for_single_neuron(
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


# -----------------------------------------------------------------------------
# Utils


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
