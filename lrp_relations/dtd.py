from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
from torch import nn
from torch.utils import hooks

from lrp_relations import local_linear

# -----------------------------------------------------------------------------
# Model


class LinearReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

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


class NetworkOutput(nn.Module):
    """Used as a marker for the output of a network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NetworkInput(nn.Module):
    """Used as a marker for the input of a network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MLP(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        output_selection: slice = slice(None, None),
    ):
        super().__init__()

        # store parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.output_selection = output_selection
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

    def slice(
        self,
        start: Union[None, NetworkInput, LinearReLU] = None,
        end: Union[None, NetworkOutput, LinearReLU] = None,
        output: Optional[slice] = None,
    ) -> MLP:
        """Returns a new MLP with only the layers between start and end."""

        if start is None or isinstance(start, NetworkInput):
            start = self.layers[0]

        if end is None or isinstance(end, NetworkOutput):
            end = self.layers[-1]

        assert isinstance(start, LinearReLU)
        assert isinstance(end, LinearReLU)

        if output is not None and self.is_final_layer(end):
            output = output
        elif output is not None and not self.is_final_layer(end):
            raise ValueError("Cannot slice output if end is not final layer")
        elif output is None and self.is_final_layer(end):
            output = self.output_selection
        else:
            output = slice(None, None)

        start_idx = self.layers.index(start)
        end_idx = self.layers.index(end)
        n_layers = end_idx - start_idx + 1
        mlp = MLP(
            n_layers,
            start.in_features,
            start.out_features,
            end.out_features,
            output,
        )
        mlp.layers = self.layers[start_idx : end_idx + 1]
        return mlp

    def get_input_with_output_greater(
        self,
        greater_as: float = 0.0,
        output: Optional[slice] = None,
        non_negative: bool = False,
        n_tries: int = 1000,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        if output is None:
            output = self.output_selection

        for _ in range(n_tries):
            x = torch.randn(1, self.input_size)
            if non_negative:
                x = x.abs()
            logit = self(x)[:, output]
            if (logit <= greater_as).all():
                continue

            break

        if (logit <= torch.tensor(greater_as)).any():
            raise RuntimeError(
                f"Could not find input with output greater than {greater_as}"
            )
        return x

    def resolve_marker(self, layer: NETWORK_LAYER) -> LinearReLU:
        if isinstance(layer, LinearReLU):
            return layer
        elif isinstance(layer, NetworkInput):
            return self.first_layer
        elif isinstance(layer, NetworkOutput):
            return self.last_layer

    def init_weights(self) -> None:
        def weight_scale(m: nn.Module) -> None:
            for p in m.parameters():
                # to compensate for negative biases, we scale the weight
                p.data[p.data > 0] = 1.4 * p.data[p.data > 0]
            if isinstance(m, LinearReLU):
                m.linear.bias.data = -m.linear.bias.data.abs()

        self.apply(weight_scale)

    def compute_input_grad(
        self,
        x: torch.Tensor,
        first: Union[NetworkInput, LinearReLU, None] = None,
        last: Union[LinearReLU, NetworkOutput, None] = None,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        x.requires_grad_(True)
        output = self(x, first, last)
        (grad,) = torch.autograd.grad(
            output, x, torch.ones_like(output), retain_graph=True
        )
        return grad

    def forward(
        self,
        x: torch.Tensor,
        first: Union[NetworkInput, LinearReLU, None] = None,
        last: Union[LinearReLU, NetworkOutput, None] = None,
    ) -> torch.Tensor:
        if first is None or isinstance(first, NetworkInput):
            first = self.layers[0]
        if last is None or isinstance(last, NetworkOutput):
            last = self.layers[-1]

        assert isinstance(first, LinearReLU)
        assert isinstance(last, LinearReLU)

        first_seen = False

        for layer in self.layers:
            if layer == first:
                first_seen = True

            if first_seen:
                x = layer(x)
            if layer == last:
                break
        if self.is_final_layer(last):
            x = x[:, self.output_selection]
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

    @property
    def first_layer(self) -> LinearReLU:
        return self.layers[0]

    @property
    def last_layer(self) -> LinearReLU:
        return self.layers[-1]

    def is_first_layer(self, layer: NETWORK_LAYER) -> bool:
        return layer == self.layers[0]

    def is_final_layer(self, layer: Union[LinearReLU, NetworkOutput]) -> bool:
        if isinstance(layer, NetworkOutput):
            return True
        return layer == self.layers[-1]

    def get_next_layer_or_output(
        self, layer: LinearReLU
    ) -> Union[LinearReLU, NetworkOutput]:
        if self.is_final_layer(layer):
            return NetworkOutput()
        return self.layers[self.layers.index(layer) + 1]

    def get_next_layer(self, layer: LinearReLU) -> LinearReLU:
        """Returns the next layer in the network."""
        if self.is_final_layer(layer):
            raise ValueError("Cannot get next layer of last layer.")
        return self.layers[self.layers.index(layer) + 1]

    def get_layer_index(self, layer: LinearReLU) -> int:
        return self.layers.index(layer)

    def get_layer_name(self, layer: NETWORK_LAYER) -> str:
        if isinstance(layer, LinearReLU):
            return f"layer{self.get_layer_index(layer) + 1}"
        elif isinstance(layer, NetworkOutput):
            return "output"
        elif isinstance(layer, NetworkInput):
            return "input"
        else:
            raise ValueError(f"Unknown layer type {type(layer)}.")

    def get_prev_layer(self, layer: NETWORK_LAYER) -> LinearReLU:
        """Returns the previous layer in the network."""

        if isinstance(layer, NetworkOutput):
            return self.layers[-1]
        else:
            if isinstance(layer, NetworkInput) or self.is_first_layer(layer):
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


def get_latex_rule_name(rule: RULE) -> str:
    latex_rule_names = {
        "w2": "$w^2$",
        "z+": "$z^+$",
        "0": "LRP$_{0}$",
    }
    if rule.name == "gamma":
        return f"$\\gamma={rule.gamma}$"
    else:
        return latex_rule_names[rule.name]


class rules:
    pinv = Rule("pinv")
    z_plus = Rule("z+")
    w2 = Rule("w2")
    x = Rule("x")
    zB = Rule("zB")
    zero = Rule("0")

    @staticmethod
    def gamma(gamma: float) -> GammaRule:
        return GammaRule(gamma=gamma)


def is_layer_rule(rule: RULE) -> bool:
    """Returns whether the rule is a layer rule."""
    return Rule(rule).name == "pinv"


def compute_root_for_layer(
    x: torch.Tensor,
    layer: LinearReLU,
    rule: RULE = "pinv",
    relevance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    w = layer.linear.weight  # [out, in]
    b = layer.linear.bias  # [out]

    if rule_name(rule) == "pinv":
        if relevance is not None:
            assert len(relevance) == 1
            return (torch.linalg.pinv(w) @ relevance[0]).unsqueeze(0)
        else:
            return (torch.linalg.pinv(w) @ b).unsqueeze(0)
    else:
        raise ValueError(f"Unknown rule {rule}")


def compute_root_for_single_neuron(
    x: torch.Tensor,
    layer: LinearReLU,
    j: int,
    rule: RULE = "z+",
    relevance: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return the DTD root point.

    Args:
        x: Input tensor.
        layer: Layer to compute the root point.
        j: Index of which neuron to compute the root point.
        rule: Rule to compute the root point (supported: `z+`, `w2`, and
            `gamma`).

    Returns:
        A root point `r` with the property `layer(r)[j] == 0`.
    """
    w = layer.linear.weight  # [out, in]
    b = layer.linear.bias  # [out]
    # x -> [b, in]

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
        assert isinstance(rule, GammaRule)
        v = x * (1 + rule.gamma * indicator_w_j_pos)
    else:
        raise ValueError()
    assert torch.allclose(
        layer.linear(x)[:, j].unsqueeze(1), x @ w_j.t() + b_j, atol=1e-3
    )

    if (v @ w_j.t()).abs().sum() <= 1e-5:
        return None

    if relevance is None:
        rel_output = (x @ w_j.t() + b_j)[:, 0]
    else:
        rel_output = relevance[:, j]

    #  This is equation (4) in DTD Appendix
    t = rel_output / (v @ w_j.t()).sum(1)
    # print("xw", (x @ w_j.t() + b_j).sum(1))
    # print("vw", (v @ w_j.t()).sum(1))
    # assert t.isnan().sum() == 0

    t[~t.isfinite()] = 0.0

    root_point = x - t.unsqueeze(1) * v
    return root_point


@dataclasses.dataclass(frozen=True)
class RecursiveRoot:
    root: torch.Tensor
    input: torch.Tensor
    layer: LinearReLU
    outputs_of_root: dict[nn.Module, torch.Tensor]
    outputs_of_input: dict[nn.Module, torch.Tensor]
    rule: Rule
    explained_neuron: Union[int, slice]
    explained_logit: int
    upper_layers: list[RecursiveRoot]

    def select(self, layer: LinearReLU) -> "RecursiveRoot":
        """Select the root point of the given layer."""
        if layer is self.layer:
            return self
        else:
            for upper_layer in self.upper_layers:
                return upper_layer.select(layer)
        raise KeyError(f"Layer {layer} not found")


@dataclasses.dataclass(frozen=True)
class RootPoint:
    root: torch.Tensor
    input: torch.Tensor
    layer: NETWORK_LAYER
    root_finder: RootFinder
    explained_neuron: Union[int, slice]
    relevance: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"RootPoint(root={self.root}, input={self.input},"
            f" root_finder={self.root_finder})"
        )


class RootFinder(abc.ABC):
    @abc.abstractmethod
    def get_root_points_for_layer(
        self,
        for_input_to_layer: NETWORK_LAYER,
        input: torch.Tensor,
        relevance_fn: RelevanceFn[REL],
    ) -> list[RootPoint]:
        """Return the root points for the given layer.

        Args:
            for_input_to_layer: The layer to find the root points of.
            input: The input to the layer.
            relevance_fn: A function that maps the output of the layer to a
                relevance score.
        """


@dataclasses.dataclass(frozen=True)
class LocalSegmentsCacheKey:
    input_layer: int
    lower_rel_layer: int
    upper_rel_layer: int
    grad_hash: int
    hash_decimals: int

    @staticmethod
    def from_rel_fn(
        rel_fn: RelevanceFn, grad_hash: int, hash_decimals: int
    ) -> LocalSegmentsCacheKey:
        mlp = rel_fn.mlp

        def get_index(layer: NETWORK_LAYER) -> int:
            return mlp.get_layer_index(mlp.resolve_marker(layer))

        return LocalSegmentsCacheKey(
            get_index(rel_fn.get_input_layer()),
            get_index(rel_fn.get_lower_rel_layer()),
            get_index(rel_fn.get_upper_rel_layer()),
            grad_hash,
            hash_decimals,
        )


@dataclasses.dataclass
class SampleRoots(RootFinder):
    model: MLP  # The model

    use_cache: bool = True
    cache: dict[
        LocalSegmentsCacheKey,
        list[RootPoint],
    ] = dataclasses.field(default_factory=dict, init=False, hash=False)
    cache_hits: int = dataclasses.field(default=0, init=False, hash=False)
    cache_misses: int = dataclasses.field(default=0, init=False, hash=False)

    use_candidates_cache: bool = False
    candidates_cache: dict[
        LocalSegmentsCacheKey,
        torch.Tensor,
    ] = dataclasses.field(default_factory=dict, init=False, hash=False)
    cache_grad_hash_decimals: int = 5

    def __post_init__(self):
        pass

    def clear_caches(self) -> None:
        self.candidates_cache.clear()
        self.cache.clear()

    def get_cache_key(
        self,
        for_input_to_layer: NETWORK_LAYER,
        input: torch.Tensor,
        relevance_fn: RelevanceFn,
    ) -> Optional[LocalSegmentsCacheKey]:
        assert not isinstance(for_input_to_layer, (NetworkOutput, NetworkInput))

        if not self.use_cache:
            return None
        grad = self.model.compute_input_grad(input, for_input_to_layer)
        grad_query = torch.round(grad, decimals=self.cache_grad_hash_decimals)
        grad_bytes = grad_query.detach().cpu().numpy().tobytes()
        grad_hash = hash(grad_bytes)

        return LocalSegmentsCacheKey.from_rel_fn(
            relevance_fn, grad_hash, self.cache_grad_hash_decimals
        )

    @abc.abstractmethod
    def sample_candidates(
        self, model: MLP, input: torch.Tensor, relevance_fn: RelevanceFn
    ) -> torch.Tensor:
        """Return a tensor of candidate root points."""

    def get_root_points_for_layer(
        self,
        for_input_to_layer: NETWORK_LAYER,
        input: torch.Tensor,
        relevance_fn: RelevanceFn,
    ) -> list[RootPoint]:
        assert not isinstance(for_input_to_layer, NetworkOutput)
        model = self.model.slice(start=for_input_to_layer)

        cache_key = self.get_cache_key(for_input_to_layer, input, relevance_fn)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        else:
            print(f"Missed cache@{len(self.cache)} with: ", cache_key)

            self.cache_misses += 1

        root_candidates = self.sample_candidates(model, input, relevance_fn)
        if cache_key is not None:
            self.candidates_cache[cache_key] = root_candidates

        candidates_rel = torch.cat(
            [
                relevance_fn(root_candidate.unsqueeze(0)).relevance
                for root_candidate in root_candidates
            ]
        )

        # rel.relevacne   [n_candidates, n_neurons]
        idx = min
        lowest_rel = candidates_rel.argmin(dim=0)
        roots = []
        for j, idx in enumerate(lowest_rel):
            roots.append(
                RootPoint(
                    root=root_candidates[idx].unsqueeze(0),
                    input=input,
                    layer=for_input_to_layer,
                    root_finder=self,
                    explained_neuron=j,
                    relevance=candidates_rel[idx],
                )
            )
        if cache_key is not None:
            self.cache[cache_key] = roots
        return roots


@dataclasses.dataclass
class InterpolationRootFinder(SampleRoots):
    args: local_linear.InterpolationArgs = dataclasses.field(
        default_factory=local_linear.InterpolationArgs,
    )

    def __repr__(self) -> str:
        return f"InterpolationRootFinder(args={self.args})"

    def sample_candidates(
        self, model: MLP, input: torch.Tensor, relevance_fn: RelevanceFn
    ) -> torch.Tensor:
        result = local_linear.sample_interpolation(model, input, args=self.args)
        return result.edge_points


@dataclasses.dataclass
class MetropolisHastingRootFinder(SampleRoots):
    args: local_linear.MetropolisHastingArgs = (
        local_linear.MetropolisHastingArgs()
    )

    def __repr__(self) -> str:
        return (
            f"LocalSegmentRoots(n_chains={self.args.n_chains}, "
            f"n_steps={self.args.n_steps}, n_warmup={self.args.n_warmup})"
        )

    def sample_candidates(
        self, model: MLP, input: torch.Tensor, relevance_fn: RelevanceFn
    ) -> torch.Tensor:
        chain = local_linear.sample_metropolis_hasting(
            model, input, args=self.args
        )
        return chain.all_samples()


@dataclasses.dataclass(frozen=True)
class LinearDTDRootFinder(RootFinder):
    """Find the roots of a neural network."""

    mlp: MLP
    explained_output: int
    rule: Rule

    def __repr__(self) -> str:
        return (
            f"RecursiveRoots(explained_output={self.explained_output},"
            f" rule={self.rule})"
        )

    def get_root_points_for_layer(
        self,
        for_input_to_layer: NETWORK_LAYER,
        input: torch.Tensor,
        relevance_fn: RelevanceFn[REL],
    ) -> list[RootPoint]:
        assert isinstance(for_input_to_layer, LinearReLU)
        assert relevance_fn.get_input_layer() == for_input_to_layer

        rel = relevance_fn(input)
        relevance = rel.relevance

        rule = getattr(for_input_to_layer, "dtd_rule", None) or self.rule
        if is_layer_rule(rule):
            roots: list[Optional[torch.Tensor]] = [
                compute_root_for_layer(
                    input, for_input_to_layer, rule, relevance
                )
            ]
        else:
            explained_neurons = list(range(for_input_to_layer.out_features))
            if self.mlp.is_final_layer(for_input_to_layer):
                explained_neurons = [self.explained_output]

            roots = [
                compute_root_for_single_neuron(
                    input, for_input_to_layer, j, rule, relevance
                )
                for j in explained_neurons
            ]

        return [
            RootPoint(root, input, for_input_to_layer, self, j, relevance)
            for j, root in enumerate(roots)
            if root is not None
        ]


@dataclasses.dataclass(frozen=True)
class RecursiveRoots(RootFinder):
    """Find the roots of a neural network."""

    mlp: MLP
    explained_output: int
    rule: Rule

    def get_root_points(
        self,
        layer: LinearReLU,
        input: torch.Tensor,
        end_at: LinearReLU,
    ) -> list[RecursiveRoot]:
        rule = getattr(layer, "rule", None) or self.rule

        roots: list[Optional[torch.Tensor]]
        if is_layer_rule(rule):
            roots = [compute_root_for_layer(input, layer, rule)]

        else:
            explained_neurons = list(range(layer.out_features))
            if self.mlp.is_final_layer(layer):
                explained_neurons = [self.explained_output]

            roots = [
                compute_root_for_single_neuron(input, layer, j, rule)
                for j in explained_neurons
            ]

        rr = []
        for j, root in enumerate(roots):
            if root is None:
                continue

            if layer != end_at:
                upper_layers = self.get_root_points(
                    self.mlp.get_next_layer(layer),
                    root,
                    end_at,
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
                    explained_neuron=j if is_layer_rule(rule) else np.s_[:],
                    explained_logit=self.explained_output,
                    upper_layers=upper_layers,
                )
            )
        if not rr:
            raise ValueError("Could not find any suitable root point!")
        return rr

    def find_roots(
        self,
        input: torch.Tensor,
        start_at: Optional[LinearReLU] = None,
        end_at: Optional[LinearReLU] = None,
        is_hidden: bool = False,
    ) -> list[RecursiveRoot]:
        """Run the recursive roots algorithm.

        Args:
            input: The input to the neural network.
            start_at: The layer to start with computing roots.
            end_at: The last layer to compute roots for.
            is_hidden: Whether the input is the hidden input to the layer
                `start_at` or not.

        Returns:
            A list of root points.
        """
        if start_at is None:
            start_at = self.mlp.layers[0]

        if end_at is None:
            end_at = self.mlp.layers[-1]

        assert isinstance(start_at, LinearReLU)
        assert isinstance(end_at, LinearReLU)

        if self.mlp.is_first_layer(start_at) or is_hidden:
            hidden = input
        else:
            hidden = self.mlp(input, last=self.mlp.get_prev_layer(start_at))

        return self.get_root_points(start_at, hidden, end_at)


# ------------------------------------------------------------------------------
# Recursive Computation of Relevances


# def view_of_relevance_fn(
#     mlp: MLP,
#     start_at: LinearReLU,
#     explained_neuron: int,
#     explained_logit: int,
#     relevance_fn: LinearReLU,
# ) -> Callable[[torch.Tensor], torch.Tensor]:
#     def relevance_fn_view(root: torch.Tensor) -> torch.Tensor:
#         output = mlp(root, first=start_at)
#         rel = relevance_fn(output[:, explained_logit].unsqueeze(1))
#         return rel[:, explained_neuron].unsqueeze(1)
#
#     return relevance_fn_view
#


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Eventually unsqueeze the tensor to a 2D tensor."""
    if x.ndim == 1:
        return x.unsqueeze(1)
    return x


NEURON = Union[int, slice]

REL = TypeVar("REL", bound="Relevance")
REL_FN = TypeVar("REL_FN", bound="RelevanceFn")


@dataclasses.dataclass(frozen=True)
class Relevance(Generic[REL_FN]):
    """A relevance score.

    Parameters:
        relevance: The relevance score.
        computed_with_fn: The relevance function that was used to compute the
            relevance.

    """

    relevance: torch.Tensor
    computed_with_fn: REL_FN

    def collect_relevances(self) -> Sequence[Relevance]:
        return [self]


NETWORK_LAYER = Union[LinearReLU, NetworkOutput, NetworkInput]


# @dataclasses.dataclass(frozen=True)
class RelevanceFn(Generic[REL], metaclass=abc.ABCMeta):
    """The abstract interface of a relevance function.

    A relevance function decomposes the relevance of the **output** of the upper
    layer to input of the lower layer. The `input` to `__call__` must be an
    activation of the layer returned from the `get_input_layer()` method.

    You can think of it as a function: $R_l^u(a_i)$ that takes the activation at
    the input layer $i$ and computes the relevance at the lower layer $l$ using
    the relevances of the upper layer $u$.
    """

    def __init__(self, mlp: MLP):
        self.mlp = mlp

    @abc.abstractmethod
    def get_input_layer(self) -> NETWORK_LAYER:
        """The layer that is used as input to the relevance function."""

    @abc.abstractmethod
    def get_lower_rel_layer(self) -> NETWORK_LAYER:
        """Returns the layer for which the relevance is decomposed."""

    @abc.abstractmethod
    def get_upper_rel_layer(self) -> NETWORK_LAYER:
        """Returns the layer for which **output** relevance is used."""

    @abc.abstractmethod
    def __call__(self, input: torch.Tensor) -> REL:
        """Compute the relevance of the input."""


@dataclasses.dataclass(frozen=True)
class RelevanceFixedLayersFn(RelevanceFn[REL], Generic[REL]):

    mlp: MLP
    input_layer: NETWORK_LAYER
    lower_rel_layer: NETWORK_LAYER
    upper_rel_layer: NETWORK_LAYER

    def get_input_layer(self) -> NETWORK_LAYER:
        return self.input_layer

    def get_lower_rel_layer(self) -> NETWORK_LAYER:
        """The input layer."""
        return self.lower_rel_layer

    def get_upper_rel_layer(self) -> NETWORK_LAYER:
        return self.upper_rel_layer

    def __call__(self, input: torch.Tensor) -> REL:
        """Compute the relevance of the input."""
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class ConstantRelFn(RelevanceFixedLayersFn[Relevance]):
    """A constant relevance function."""

    relevance: torch.Tensor

    def __call__(self, input: torch.Tensor) -> Relevance:
        del input
        return Relevance(self.relevance, self)


@dataclasses.dataclass(frozen=True)
class NetworkOutputRelevanceFn(RelevanceFixedLayersFn[Relevance]):
    """A relevance function that computes the relevance of a network output."""

    mlp: MLP
    explained_output: NEURON
    input_layer: NETWORK_LAYER
    lower_rel_layer: NETWORK_LAYER = dataclasses.field(init=False)
    upper_rel_layer: NetworkOutput = dataclasses.field(
        init=False, default=NetworkOutput()
    )

    def __post_init__(self):
        # because of frozen=True, we need to use __setattr__
        object.__setattr__(self, "lower_rel_layer", self.input_layer)

    def __call__(self, input: torch.Tensor) -> Relevance:
        output = self.mlp(input, self.lower_rel_layer)[:, self.explained_output]
        output = ensure_2d(output)
        return Relevance(output, self)


@dataclasses.dataclass(frozen=True)
class OutputRel(Relevance[REL_FN], Generic[REL_FN]):
    def __repr__(self) -> str:
        return f"OutputRel(relevance={self.relevance})"


@dataclasses.dataclass(frozen=True)
class OutputRelFn(RelevanceFixedLayersFn[OutputRel]):
    mlp: MLP
    explained_output: NEURON

    input_layer: NetworkOutput = dataclasses.field(
        default=NetworkOutput(), init=False
    )
    lower_rel_layer: NetworkOutput = dataclasses.field(
        default=NetworkOutput(), init=False
    )
    upper_rel_layer: NetworkOutput = dataclasses.field(
        default=NetworkOutput(), init=False
    )

    def __repr__(self) -> str:
        return f"OutputRelFn(explained_output={self.explained_output})"

    def __call__(self, input: torch.Tensor) -> "OutputRel[OutputRelFn]":
        rel = input[:, self.explained_output]
        return OutputRel(ensure_2d(rel), self)


@dataclasses.dataclass(frozen=True)
class FullBackwardRel(Relevance):
    roots: list[RootPoint]
    relevance: torch.Tensor
    roots_relevance: list[torch.Tensor]
    unresolved_relevance: list[torch.Tensor]
    root_grads: list[torch.Tensor]
    relevance_upper_layers: list[Relevance]

    def __repr__(self) -> str:
        return (
            f"FullBackwardRel(#roots={len(self.roots)}, "
            f"relevance_upper_layers={self.relevance_upper_layers}, "
            f"relevance={self.relevance}, "
            f"unresolved_relevance={self.unresolved_relevance})"
        )

    def collect_relevances(self) -> Sequence[Relevance]:
        self_rel: list[Relevance] = [self]
        upper_rel: list[Relevance] = []
        for rel in self.relevance_upper_layers:
            upper_rel.extend(rel.collect_relevances())

        return self_rel + upper_rel

    def _collect_info(self, callgraph: list[str]) -> list[dict[str, Any]]:
        mlp = self.computed_with_fn.mlp
        rel_input = self.computed_with_fn.get_input_layer()
        if isinstance(rel_input, LinearReLU):
            layer_name = mlp.get_layer_name(rel_input)
        else:
            layer_name = "output"

        callgraph_w_layer = callgraph + [layer_name]
        data = []

        if isinstance(self, FullBackwardRel):
            for root, r, rel_unresolved, rel_decomposed in zip(
                self.roots,
                self.relevance_upper_layers,
                self.unresolved_relevance,
                self.roots_relevance,
            ):
                j = root.explained_neuron
                callgraph_w_root = callgraph_w_layer + [f"root_{j}"]

                assert root.relevance is not None

                root_logit = mlp(root.root, first=root.layer)
                input_logit = mlp(root.input, first=root.layer)
                data.append(
                    {
                        "layer": layer_name,
                        "unresolved_relevance": rel_unresolved.detach().numpy(),
                        "relevance": rel_decomposed.detach().numpy(),
                        "callgraph": callgraph_w_root,
                        "root": root.root.detach().numpy(),
                        "input": root.input.detach().numpy(),
                        "root_logit": root_logit.detach().item(),
                        "input_logit": input_logit.detach().item(),
                        "explained_neuron": root.explained_neuron,
                        "root_relevance": root.relevance.detach().numpy(),
                    }
                )
                if isinstance(r, FullBackwardRel):
                    data.extend(r._collect_info(callgraph_w_root))
        return data

    def collect_info(self):
        return self._collect_info([])


@dataclasses.dataclass(frozen=True)
class StabilizeGradient:
    noise_std: float = 1e-6
    max_tries: int = 3


@dataclasses.dataclass(frozen=True)
class ForwardInputRelFn(RelevanceFn[REL], Generic[REL, REL_FN]):
    """A wrapper around a relevance function that takes care that the input is
    forwarded to the input layer of the relevance function."""

    mlp: MLP
    input_layer: NETWORK_LAYER
    wrapped_rel_fn: RelevanceFn[REL]

    def get_input_layer(self) -> NETWORK_LAYER:
        return self.input_layer

    def get_upper_rel_layer(self) -> NETWORK_LAYER:
        return self.wrapped_rel_fn.get_upper_rel_layer()

    def get_lower_rel_layer(self) -> NETWORK_LAYER:
        return self.wrapped_rel_fn.get_lower_rel_layer()

    def __call__(self, input: torch.Tensor) -> REL:
        forwarded_input = self.mlp(
            input,
            self.input_layer,
            self.mlp.get_prev_layer(self.wrapped_rel_fn.get_input_layer()),
        )
        return self.wrapped_rel_fn(forwarded_input)


@dataclasses.dataclass(frozen=True)
class FullBackwardFn(RelevanceFixedLayersFn[FullBackwardRel]):
    mlp: MLP
    input_layer: LinearReLU
    upper_rel_layer: NETWORK_LAYER
    relevance_fn: RelevanceFn
    root_finder: RootFinder
    check_nans: bool = True
    stabilize_grad: Optional[StabilizeGradient] = StabilizeGradient()
    lower_rel_layer: LinearReLU = dataclasses.field(init=False)

    def __post_init__(self):
        # because of frozen=True, we need to use __setattr__
        object.__setattr__(self, "lower_rel_layer", self.input_layer)

    @staticmethod
    def get_fn_builder(
        mlp: MLP,
        root_finder: RootFinder,
        check_nans: bool = True,
        stabilize_grad: Optional[StabilizeGradient] = StabilizeGradient(),
    ) -> REL_FN_FACTORY:
        def factory(
            input_layer: NETWORK_LAYER, upper_rel_fn: RelevanceFn
        ) -> FullBackwardFn:
            assert isinstance(input_layer, LinearReLU)
            return FullBackwardFn(
                mlp=mlp,
                input_layer=input_layer,
                upper_rel_layer=upper_rel_fn.get_lower_rel_layer(),
                relevance_fn=upper_rel_fn,
                root_finder=root_finder,
                check_nans=check_nans,
                stabilize_grad=stabilize_grad,
            )

        return factory

    def __repr__(self) -> str:
        to_idx = self.mlp.get_layer_name(self.input_layer)
        from_idx = self.mlp.get_layer_name(self.upper_rel_layer)

        return (
            f"FullBackwardFn(to_input_of={to_idx}, "
            f"from_output_of={from_idx}, "
            f"rule={self.root_finder}, ...)"
        )

    def __call__(self, input: torch.Tensor) -> FullBackwardRel:
        def get_grad_and_rel(
            root: RootPoint,
        ) -> tuple[torch.Tensor, torch.Tensor, Relevance]:
            # output = self.mlp(root.root, self.input_layer,
            # self.upper_rel_layer)
            rel_info_from = forward_rel_fn(root.root)
            rel_from = rel_info_from.relevance
            rel_from = ensure_2d(rel_from[:, root.explained_neuron])

            (grad_root,) = torch.autograd.grad(
                rel_from.sum(), root.root, retain_graph=True
            )
            return grad_root, rel_from, rel_info_from

        assert input.shape[0] == 1
        input.requires_grad_(True)

        forward_rel_fn = ForwardInputRelFn[Relevance, RelevanceFn](
            self.mlp, self.input_layer, self.relevance_fn
        )
        input_rel = forward_rel_fn(input)

        roots = self.root_finder.get_root_points_for_layer(
            self.input_layer, input, relevance_fn=forward_rel_fn
        )

        roots_relevance = []
        rel_infos = []
        unresolved_rel = []
        root_grads = []
        for root in roots:
            root_point = root.root.clone()
            root.root.requires_grad_(True)

            input_rel_j = input_rel.relevance[:, root.explained_neuron]
            if input_rel_j.sum() <= 0:
                # assert len(input_rel_j) == 1
                # no relevance for this neuron, so we can skip it
                continue

            # in case of nans, we stabilize the gradient by adding a little
            # Gaussian noise to the root point.
            i = 0
            while True:
                if i != 0 and self.stabilize_grad is not None:
                    noise_std = self.stabilize_grad.noise_std
                    noise = noise_std * torch.randn_like(root.root)
                    root.root[:] = root_point + noise

                grad_root, unresolved_rel_j, rel_info_from = get_grad_and_rel(
                    root
                )

                has_nans = not torch.isfinite(grad_root).all()
                grad_zero = grad_root.abs().sum() < 1e-7

                # In case: grad is zero or nan --> try again
                if self.stabilize_grad is not None and (has_nans or grad_zero):
                    if i + 1 <= self.stabilize_grad.max_tries:
                        i += 1
                        continue

                if self.check_nans and not torch.isfinite(grad_root).all():
                    raise RuntimeError("Gradient of root is not finite")
                break

            rel_j = grad_root * (input - root.root)

            # assert grad_root.abs().sum() > 1e-7
            # assert root.relevance is not None
            unresolved_rel.append(unresolved_rel_j)
            if root.relevance is not None and False:
                # f(x)  = f(x̃) + ∇f(x̃) (x - x̃)
                # input_rel == unresolved_rel_j + rel_j.sum()
                assert torch.allclose(
                    input_rel_j,
                    unresolved_rel_j + rel_j.sum(1),
                    atol=1e-3,
                    rtol=0.10,
                )
            roots_relevance.append(rel_j)
            rel_infos.append(rel_info_from)
            root_grads.append(grad_root)

        # to_idx = self.mlp.get_layer_name(self.input_layer)
        # from_idx = self.mlp.get_layer_name(self.upper_rel_layer)
        # print(f"Finished decomposing: {to_idx} <- {from_idx}")

        total_rel = torch.stack(roots_relevance, dim=0).sum(0)
        return FullBackwardRel(
            roots=roots,
            relevance=total_rel,
            roots_relevance=roots_relevance,
            relevance_upper_layers=rel_infos,
            unresolved_relevance=unresolved_rel,
            root_grads=root_grads,
            computed_with_fn=self,
        )


@dataclasses.dataclass(frozen=True)
class TrainFreeRel(Relevance, Generic[REL_FN]):
    roots: list[RootPoint]
    grad_roots: list[torch.Tensor]
    roots_relevance: list[torch.Tensor]
    relevance_of_upper_layer: Relevance[REL_FN]
    relevance: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"TrainFreeRel(#roots={len(self.roots)}, "
            f"relevance_of_upper_layer={self.relevance_of_upper_layer}, "
            f"relevance={self.relevance}, ...)"
        )

    def collect_relevances(self) -> Sequence[Relevance]:
        self_rel: list[Relevance] = [self]
        upper_rel: Sequence[
            Relevance[REL_FN]
        ] = self.relevance_of_upper_layer.collect_relevances()

        return self_rel + list(upper_rel)


@dataclasses.dataclass(frozen=True)
class TrainFreeFn(RelevanceFixedLayersFn[TrainFreeRel]):
    mlp: MLP
    input_layer: NETWORK_LAYER
    upper_rel_layer: NETWORK_LAYER
    relevance_fn: RelevanceFn
    root_finder: RootFinder
    check_nans: bool = True
    check_consistent: bool = True
    # whether to omit the ReLU activation, when computing the gradient
    omit_relu: bool = True
    lower_rel_layer: LinearReLU = dataclasses.field(init=False)

    @staticmethod
    def get_fn_builder(
        mlp: MLP,
        root_finder: RootFinder,
        check_nans: bool = True,
        check_consistent: bool = True,
    ) -> REL_FN_FACTORY:
        def factory(
            input_layer: NETWORK_LAYER, upper_rel_fn: RelevanceFn
        ) -> TrainFreeFn:
            return TrainFreeFn(
                mlp=mlp,
                input_layer=input_layer,
                upper_rel_layer=upper_rel_fn.get_lower_rel_layer(),
                relevance_fn=upper_rel_fn,
                root_finder=root_finder,
                check_nans=check_nans,
                check_consistent=check_consistent,
            )

        return factory

    def __post_init__(self):
        # because of frozen=True, we need to use __setattr__
        object.__setattr__(self, "lower_rel_layer", self.input_layer)

    def __repr__(self) -> str:
        to_idx = self.mlp.get_layer_name(self.input_layer)
        from_idx = self.mlp.get_layer_name(self.upper_rel_layer)

        return (
            f"TrainFreeFn(to_input_of={to_idx}, "
            f"from_output_of={from_idx}, "
            f"rule={self.root_finder}, ...)"
        )

    def __call__(self, input: torch.Tensor) -> TrainFreeRel:
        assert not isinstance(self.input_layer, NetworkOutput)
        input_layer = self.mlp.resolve_marker(self.input_layer)

        # Approximation: use input instead of root
        rel_info_from = self.relevance_fn(input_layer(input))
        rel_from = rel_info_from.relevance

        const_rel = ConstantRelFn(
            self.mlp,
            input_layer,
            self.lower_rel_layer,
            self.relevance_fn.get_upper_rel_layer(),
            rel_from,
        )
        roots = self.root_finder.get_root_points_for_layer(
            input_layer,
            input,
            const_rel,
        )
        roots_relevance = []
        rel_infos = []
        grad_roots = []

        for root in roots:
            root.root.requires_grad_(True)

            if self.omit_relu:
                # To ensure gradient is not zero, only derive linear layer
                output = input_layer.linear(root.root)
            else:
                output = input_layer(root.root)

            output_sel = ensure_2d(output[:, root.explained_neuron])

            (grad_root,) = torch.autograd.grad(
                output_sel,
                root.root,
                # grad_outputs=ensure_2d(rel_from[:, root.explained_neuron]),
                grad_outputs=torch.ones_like(output_sel),
            )

            rel_j = grad_root * (input - root.root)

            if self.check_consistent:
                assert torch.allclose(
                    rel_j.sum(), rel_from[0, root.explained_neuron], atol=1e-7
                )
            # TODO: check nans

            roots_relevance.append(rel_j)
            rel_infos.append(rel_info_from)
            grad_roots.append(grad_root)

        # to_idx = self.mlp.get_layer_name(input_layer)
        # from_idx = self.mlp.get_layer_name(self.upper_rel_layer)
        # print(f"Finished decomposing: {to_idx} <- {from_idx}")
        total_rel = torch.stack(roots_relevance, dim=0).sum(0)
        # assert (total_rel >= 0).all()
        return TrainFreeRel(
            roots=roots,
            grad_roots=grad_roots,
            roots_relevance=roots_relevance,
            relevance_of_upper_layer=rel_info_from,
            relevance=total_rel,
            computed_with_fn=self,
        )


REL_FN_FACTORY = Callable[[NETWORK_LAYER, RelevanceFn], RelevanceFn]


def get_decompose_relevance_fns(
    mlp: MLP,
    explained_output: NEURON,
    relevance_builder: REL_FN_FACTORY,
    # rule: Optional[Rule] = None,
    # root_finder: Optional[RootFinder] = None,
) -> list[RelevanceFn[Relevance]]:
    rel_fns: list[RelevanceFn] = [
        OutputRelFn(
            mlp=mlp,
            explained_output=explained_output,
        )
    ]
    # if root_finder is None:
    #     if rule is None:
    #         raise ValueError("Either rule or root_finder must be specified")
    #     root_finder = RecursiveRoots(
    #         mlp=mlp, explained_output=explained_output, rule=rule
    #     )

    for layer in reversed(mlp.layers[:]):
        rel_fns.append(relevance_builder(layer, rel_fns[-1]))

    return rel_fns


# ------------------------------------------------------------------------------


def get_relevance_hidden(
    net: TwoLayerMLP,
    x: torch.Tensor,
    j: int = 0,
    rule: RULE = "z+",
) -> torch.Tensor:
    return get_relevance_hidden_and_root(net, x, j, rule)[0]


def get_relevance_hidden_and_root(
    net: TwoLayerMLP,
    x: torch.Tensor,
    j: int = 0,
    rule: RULE = "z+",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the relevance of each neuron in the hidden layer and the
    corresponding root point.

    This applies the given DTD rule to the final layer.

    Args:
        net: TwoLayerMLP network.
        x: Input tensor.
        j: Index of which neuron in the final layer to compute the relevance.
        rule: Rule to compute the root point

    Returns:
        A tuple of two tensors: (the relevance of each neuron in the hidden
            layer, the corresponding root point).
    """
    with record_all_outputs(net) as outputs:
        net(x)

    # outputs
    hidden = outputs[net.layer1][0]
    hidden_root = compute_root_for_single_neuron(
        hidden, net.layer2, j, rule=rule
    )
    assert hidden_root is not None
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
