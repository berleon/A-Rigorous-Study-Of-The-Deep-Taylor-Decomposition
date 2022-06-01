from __future__ import annotations

import abc
import dataclasses
import itertools
from typing import Callable, Generic, Optional, TypeVar, Union

import numpy as np
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


class NetworkOutput(nn.Module):
    """Used as a marker for the output of a network."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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

    def init_weights(self) -> None:
        def weight_scale(m: nn.Module) -> None:
            for p in m.parameters():
                # to compensate for negative biases, we scale the weight
                p.data[p.data > 0] = 1.4 * p.data[p.data > 0]
            if isinstance(m, LinearReLU):
                m.linear.bias.data = -m.linear.bias.data.abs()

        self.apply(weight_scale)

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

    @property
    def first_layer(self) -> LinearReLU:
        return self.layers[0]

    @property
    def last_layer(self) -> LinearReLU:
        return self.layers[-1]

    def is_first_layer(self, layer: LinearReLU) -> bool:
        return layer == self.layers[0]

    def is_final_layer(self, layer: LinearReLU) -> bool:
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
    layer: LinearReLU
    rule: Rule
    explained_neuron: Union[int, slice]
    relevance: Optional[torch.Tensor]


class RootFinder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_root_points_for_layer(
        self,
        for_input_to_layer: LinearReLU,
        input: torch.Tensor,
        relevance_fn: RelevanceFn,
    ) -> list[RootPoint]:
        """Return the root points for the given layer.

        Args:
            for_input_to_layer: The layer to find the root points of.
            input: The input to the layer.
            relevance_fn: A function that maps the output of the layer to a
                relevance score.
        """


@dataclasses.dataclass(frozen=True)
class RecursiveRoots(RootFinder):
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
        layer: LinearReLU,
        input: torch.Tensor,
        relevance_fn: RelevanceFn[REL],
    ) -> list[RootPoint]:

        rel = relevance_fn(input)
        relevance = rel.relevance

        rule = getattr(layer, "rule", None) or self.rule
        if is_layer_rule(rule):
            roots: list[Optional[torch.Tensor]] = [
                compute_root_for_layer(input, layer, rule, relevance)
            ]
        else:
            explained_neurons = list(range(layer.out_features))
            if self.mlp.is_final_layer(layer):
                explained_neurons = [self.explained_output]

            roots = [
                compute_root_for_single_neuron(input, layer, j, rule, relevance)
                for j in explained_neurons
            ]

        return [
            RootPoint(root, input, layer, rule, j, relevance)
            for j, root in enumerate(roots)
            if root is not None
        ]

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

        if self.mlp.is_first_layer(start_at) or is_hidden:
            hidden = input
        else:
            hidden = self.mlp(input, last=self.mlp.get_prev_layer(start_at))

        if end_at is None:
            end_at = self.mlp.layers[-1]
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

    def collect_relevances(self) -> list[Relevance]:
        return [self]


NETWORK_LAYER = Union[LinearReLU, NetworkOutput]


@dataclasses.dataclass(frozen=True)
class RelevanceFn(Generic[REL]):
    mlp: MLP

    @property
    def input_layer(self) -> NETWORK_LAYER:
        """The input layer."""
        raise NotImplementedError()

    @property
    def output_layer(self) -> NETWORK_LAYER:
        raise NotImplementedError()

    def __call__(self, input: torch.Tensor) -> REL:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class ConstantRelFn(RelevanceFn[Relevance]):
    """A constant relevance function."""

    relevance: torch.Tensor
    layer: NETWORK_LAYER

    @property
    def input_layer(self) -> NETWORK_LAYER:
        return self.layer

    def __call__(self, input: torch.Tensor) -> Relevance:
        del input
        return Relevance(self.relevance, self)


@dataclasses.dataclass(frozen=True)
class OutputRel(Relevance[REL_FN], Generic[REL_FN]):
    def __repr__(self) -> str:
        return f"OutputRel(relevance={self.relevance})"


@dataclasses.dataclass(frozen=True)
class OutputRelFn(RelevanceFn[OutputRel]):
    mlp: MLP
    explained_output: NEURON

    @property
    def input_layer(self) -> LinearReLU:
        return self.mlp.layers[-1]

    def __repr__(self) -> str:
        return f"OutputRelFn(explained_output={self.explained_output})"

    def __call__(self, input: torch.Tensor) -> "OutputRel[OutputRelFn]":
        rel = input[:, self.explained_output]
        return OutputRel(ensure_2d(rel), self)


@dataclasses.dataclass(frozen=True)
class DecomposedRel(Relevance):
    roots: list[RecursiveRoot]
    relevance_of_from_layer: torch.Tensor
    relevance: torch.Tensor
    roots_relevance: list[torch.Tensor]
    relevance_info_from: list[Relevance]

    def __repr__(self) -> str:
        return (
            f"DecomposedRel(#roots={len(self.roots)}, "
            f"relevance_of_from_layer={self.relevance_of_from_layer}, "
            f"relevance={self.relevance}, ...)"
        )


@dataclasses.dataclass(frozen=True)
class StabilizeGradient:
    noise_std: float = 1e-6
    max_tries: int = 3


@dataclasses.dataclass(frozen=True)
class DecomposedRelFn(RelevanceFn[DecomposedRel]):
    mlp: MLP
    to_input_of: LinearReLU
    from_output_of: LinearReLU
    relevance_fn: RelevanceFn
    root_finder: RecursiveRoots
    check_nans: bool = True
    stabilize_grad: Optional[StabilizeGradient] = StabilizeGradient()

    def __repr__(self) -> str:
        to_idx = self.mlp.layers.index(self.to_input_of)
        from_idx = self.mlp.layers.index(self.from_output_of)

        return (
            f"DecomposedRelFn(to_input_of={to_idx}, "
            f"from_output_of={from_idx}, "
            f"rule={self.root_finder.rule.name}, ...)"
        )

    def __call__(self, input: torch.Tensor) -> DecomposedRel:
        def get_grad_and_rel(
            root: RecursiveRoot,
        ) -> tuple[torch.Tensor, torch.Tensor, Relevance]:
            output = self.mlp(root.root, self.to_input_of, self.from_output_of)
            rel_info_from = self.relevance_fn(output)
            rel_from = rel_info_from.relevance
            rel_from = ensure_2d(rel_from[:, root.explained_neuron])

            (grad_root,) = torch.autograd.grad(
                rel_from.sum(), root.root, retain_graph=True
            )
            return grad_root, rel_from, rel_info_from

        input.requires_grad_(True)

        to_idx = self.mlp.get_layer_index(self.to_input_of)
        from_idx = self.mlp.get_layer_index(self.from_output_of)

        roots = self.root_finder.find_roots(
            input,
            start_at=self.to_input_of,
            end_at=self.to_input_of,
            is_hidden=True,
        )
        roots_relevance = []
        rel_infos = []
        for root in roots:
            root_point = root.root.clone()
            root.root.requires_grad_(True)
            for i in itertools.count():
                if i == 0 and self.stabilize_grad is not None:
                    noise_std = self.stabilize_grad.noise_std
                    noise = noise_std * torch.randn_like(root.root)
                    root.root[:] = root_point + noise
                grad_root, rel_from, rel_info_from = get_grad_and_rel(root)

                has_nans = not torch.isfinite(grad_root).all()
                grad_zero = grad_root.abs().sum() < 1e-7
                if self.stabilize_grad is not None and (has_nans or grad_zero):
                    if i + 1 <= self.stabilize_grad.max_tries:
                        continue

                if self.check_nans and not torch.isfinite(grad_root).all():
                    raise ValueError("Gradient of root is not finite")
                break

            rel_j = rel_from + grad_root * (input - root.root)
            roots_relevance.append(rel_j)
            rel_infos.append(rel_info_from)

        print(f"Finished decomposing: {to_idx} <- {from_idx}")
        total_rel = torch.stack(roots_relevance, dim=0).sum(0)
        return DecomposedRel(
            roots=roots,
            relevance_of_from_layer=rel_from,
            relevance=total_rel,
            roots_relevance=roots_relevance,
            relevance_info_from=rel_infos,
            computed_with_fn=self,
        )


@dataclasses.dataclass(frozen=True)
class TrainFreeRel(Relevance):
    roots: list[RootPoint]
    grad_roots: list[torch.Tensor]
    roots_relevance: list[torch.Tensor]
    relevance_of_upper_layer: Relevance
    relevance: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"TrainFreeRel(#roots={len(self.roots)}, "
            f"relevance_of_upper_layer={self.relevance_of_upper_layer}, "
            f"relevance={self.relevance}, ...)"
        )

    def collect_relevances(self) -> list[Relevance]:
        self_rel: list[Relevance] = [self]
        upper_rel: list[
            Relevance
        ] = self.relevance_of_upper_layer.collect_relevances()

        return self_rel + upper_rel


@dataclasses.dataclass(frozen=True)
class TrainFreeFn(RelevanceFn[TrainFreeRel]):
    mlp: MLP
    to_input_of: LinearReLU
    from_output_of: LinearReLU
    relevance_fn: RelevanceFn
    root_finder: RootFinder
    check_nans: bool = True
    stabilize_grad: Optional[StabilizeGradient] = StabilizeGradient()

    def __repr__(self) -> str:
        to_idx = self.mlp.layers.index(self.to_input_of)
        from_idx = self.mlp.layers.index(self.from_output_of)

        return (
            f"TrainFreeFn(to_input_of={to_idx}, "
            f"from_output_of={from_idx}, "
            f"rule={self.root_finder}, ...)"
        )

    def __call__(self, input: torch.Tensor) -> TrainFreeRel:
        to_idx = self.mlp.get_layer_index(self.to_input_of)
        from_idx = self.mlp.get_layer_index(self.from_output_of)

        # Approximation: use input instead of root
        rel_info_from = self.relevance_fn(self.to_input_of(input))
        rel_from = rel_info_from.relevance

        const_rel = ConstantRelFn(
            self.mlp,
            rel_from,
            self.mlp.get_next_layer_or_output(self.to_input_of),
        )
        roots = self.root_finder.get_root_points_for_layer(
            self.to_input_of,
            input,
            const_rel,
        )
        roots_relevance = []
        rel_infos = []
        grad_roots = []

        for root in roots:
            root.root.requires_grad_(True)

            # To ensure gradient is not zero, only derive linear layer
            output = self.to_input_of.linear(root.root)

            output_sel = ensure_2d(output[:, root.explained_neuron])

            (grad_root,) = torch.autograd.grad(
                output_sel,
                root.root,
                # grad_outputs=ensure_2d(rel_from[:, root.explained_neuron]),
                grad_outputs=torch.ones_like(output_sel),
            )

            rel_j = grad_root * (input - root.root)
            assert torch.allclose(
                rel_j.sum(), rel_from[0, root.explained_neuron]
            )
            roots_relevance.append(rel_j)
            rel_infos.append(rel_info_from)
            grad_roots.append(grad_root)

        print(f"Finished decomposing: {to_idx} <- {from_idx}")
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


def get_decompose_relevance_fns(
    mlp: MLP,
    explained_output: int,
    rule: Rule,
    root_finder: Optional[RecursiveRoots] = None,
    decomposition: str = "full",
) -> list[RelevanceFn]:
    rel_fns: list[RelevanceFn] = [
        OutputRelFn(mlp=mlp, explained_output=explained_output)
    ]
    if root_finder is None:
        root_finder = RecursiveRoots(
            mlp=mlp, explained_output=explained_output, rule=rule
        )

    for layer in reversed(mlp.layers[:]):

        if decomposition == "full":
            rel_fns.append(
                DecomposedRelFn(
                    mlp=mlp,
                    to_input_of=layer,
                    from_output_of=layer,
                    relevance_fn=rel_fns[-1],
                    root_finder=root_finder,
                )
            )
        elif decomposition == "train_free":
            rel_fns.append(
                TrainFreeFn(
                    mlp=mlp,
                    to_input_of=layer,
                    from_output_of=layer,
                    relevance_fn=rel_fns[-1],
                    root_finder=root_finder,
                )
            )

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


# -----------------------------------------------------------------------------
# Old code to randomly sample root points
#
# Problems:
#    * The sampled roots will not have the same gradient
#    * Not specific enough --> might not return roots
#


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
        input = ensure_2d(input)

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
