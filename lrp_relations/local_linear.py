import dataclasses

import einops
import torch
from torch import nn
from tqdm import auto as tqdm


def _get_start_grad(
    model: nn.Module,
    input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> torch.Tensor:
    input.requires_grad_(True)
    out = model(input)
    (grad,) = torch.autograd.grad([out], [input], torch.ones_like(out))

    assert torch.allclose(
        grad,
        grad[0].unsqueeze(0).repeat(grad.shape[0], 1),
        rtol=rtol,
        atol=atol,
    )
    return grad[:1]


@dataclasses.dataclass(frozen=True)
class MetropolisHastingArgs:
    init_scale: float = 1e-6
    n_chains: int = 10
    n_steps: int = 1000
    grad_rtol: float = 1e-3
    grad_atol: float = 1e-5
    n_warmup: int = 100
    show_progress: bool = False
    enforce_positive: bool = True


@dataclasses.dataclass
class MetropolisHastingResult:
    chain: torch.Tensor
    accept_ratio: torch.Tensor
    scaling: torch.Tensor
    start: torch.Tensor
    start_grad: torch.Tensor
    args: MetropolisHastingArgs

    def all_samples(self):
        steps, b, d = self.chain.shape
        return self.chain.view(steps * b, d)


def sample_metropolis_hasting(
    model: nn.Module,
    start: torch.Tensor,
    args: MetropolisHastingArgs = MetropolisHastingArgs(),
) -> MetropolisHastingResult:
    current = start

    start_grad = _get_start_grad(model, start, args.grad_rtol, args.grad_atol)

    scale = args.init_scale * current.new_ones(1, start.shape[1])
    chain = []
    accept_ratio = []
    scale_history = []

    proposals = current.clone().detach()
    noise = current.clone().detach()  # avoid allocating memory
    for i in tqdm.trange(
        args.n_warmup + args.n_steps, disable=not args.show_progress
    ):
        # proposals = (current + scale * torch.randn_like(current))
        torch.randn(noise.shape, out=noise)
        proposals.requires_grad_(False)
        proposals.copy_(current.detach()).add_(noise.mul_(scale.detach()))
        if args.enforce_positive:
            proposals.clamp_(min=0)
        proposals.requires_grad_(True)
        out = model(proposals)
        (grad,) = torch.autograd.grad([out], [proposals], torch.ones_like(out))
        # f(x') / f(x)
        accept = (
            torch.isclose(
                start_grad,
                grad,
                args.grad_rtol,
                args.grad_atol,
            )
            .all(dim=1, keepdim=True)
            .float()
        )
        # Proportional controll
        # if accept > 0.9:
        #     increase the scale

        # if i % 50 == 0:
        #     scale = max(1 ** (-i), 0.001) * current.std(0, keepdim=True)

        diff = -(0.90 - accept.mean())  # .clamp(-0.1, 0.1)
        steps_to_scaling_dims = 20
        d = (
            i % (steps_to_scaling_dims * start.size(1))
        ) // steps_to_scaling_dims
        scale[:, d] = (15**diff) * scale[:, d]
        scale = torch.clamp(scale, min=1e-6, max=0.01)
        assert scale.shape == (1, start.shape[1]), scale.shape

        current = (proposals * accept + (1 - accept) * current).detach()
        if i >= args.n_warmup:
            accept_ratio.append(accept.mean())
            chain.append(current)
            scale_history.append(scale.clone())
    return MetropolisHastingResult(
        torch.stack(chain),
        torch.stack(accept_ratio),
        torch.cat(scale_history),
        start,
        start_grad,
        args,
    )


@dataclasses.dataclass
class InterpolationArgs:
    batch_size: int = 50
    n_batches: int = 1
    init_scale: float = 1
    n_interpolation_points: int = 10
    n_refinement_steps: int = 10
    grad_rtol: float = 1e-3
    grad_atol: float = 1e-5
    show_progress: bool = False
    enforce_non_negative: bool = True


@dataclasses.dataclass
class InterpolationResults:
    all_valid_points: torch.Tensor
    edge_points: torch.Tensor
    edge_tolerance: torch.Tensor
    args: InterpolationArgs


def sample_interpolation(
    model: nn.Module,
    input: torch.Tensor,
    args: InterpolationArgs = InterpolationArgs(),
) -> InterpolationResults:
    all_valid_points = []
    proposals = input.clone().detach()

    input_grad = _get_start_grad(model, input, args.grad_rtol, args.grad_atol)

    if input.shape[0] == 1:
        input_batch = input.repeat(args.batch_size, 1)
    else:
        input_batch = input

    if args.enforce_non_negative and (input_batch.min() < 0).any():
        raise ValueError(
            "Should enforce non negative sample but input is negative!"
        )
    n = input_batch.shape[0]
    d = input.shape[1]
    linspace_10 = 1 - torch.linspace(
        0, 1, steps=args.n_interpolation_points
    ).view(1, -1, 1).repeat(n, 1, d)

    edge_points: torch.Tensor
    edge_tolerance: torch.Tensor
    for i in tqdm.trange(args.n_batches, disable=not args.show_progress):

        directions = torch.randn_like(input_batch)
        directions /= directions.norm(dim=1, keepdim=True)
        directions = directions.unsqueeze(1)

        low = torch.zeros_like(directions)
        high = args.init_scale * torch.ones_like(directions)

        for step in range(args.n_refinement_steps):
            scale = linspace_10 * (high - low) + low

            proposals = input_batch.unsqueeze(1) + scale * directions

            if step == 0:
                assert torch.allclose(proposals[:, -1], input_batch)

            assert proposals.shape == (n, args.n_interpolation_points, d)

            if args.enforce_non_negative:
                proposals.clamp_(min=0)

            proposals_flat = einops.rearrange(proposals, "n i d -> (n i) d")
            proposals_flat.requires_grad_(True)

            assert proposals_flat.shape == (n * args.n_interpolation_points, d)
            out = model(proposals_flat)

            (grad,) = torch.autograd.grad(
                [out], [proposals_flat], torch.ones_like(out)
            )
            input_grad_repeated = input_grad.repeat(len(grad), 1)

            accept = torch.isclose(
                input_grad_repeated,
                grad,
                args.grad_rtol,
                args.grad_atol,
            ).all(dim=1)

            all_valid_points.append(proposals_flat[accept])

            accept = einops.rearrange(
                accept, "(n i) -> n i ", i=args.n_interpolation_points
            )

            # ensure the lowest points are always accepted
            assert accept[:, -1].all()

            # get the first valid point
            idx = accept.float().argmax(dim=1)

            edge_points = proposals[torch.arange(len(idx)), idx]
            edge_tolerance = high - low
            low = scale[idx]
            high = scale[idx + 1]

    result = InterpolationResults(
        torch.cat(all_valid_points),
        edge_points,
        edge_tolerance,
        args,
    )
    assert result.all_valid_points.shape[0] >= 1
    return result
