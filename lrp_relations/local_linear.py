import dataclasses

import savethat
import torch
from torch import nn
from tqdm import auto as tqdm

from lrp_relations import dtd


@dataclasses.dataclass
class SamplingResult:
    chain: torch.Tensor
    accept_ratio: torch.Tensor
    scaling: torch.Tensor
    start: torch.Tensor
    start_grad: torch.Tensor
    # parameters
    init_scale: float
    n_steps: int
    grad_rtol: float
    grad_atol: float
    n_warmup: int

    def all_samples(self):
        steps, b, d = self.chain.shape
        return self.chain.view(steps * b, d)


def sample(
    model: nn.Module,
    start: torch.Tensor,
    init_scale: float = 1e-6,
    n_steps: int = 1000,
    grad_rtol: float = 1e-3,
    grad_atol: float = 1e-5,
    n_warmup: int = 100,
    show_progress: bool = False,
) -> SamplingResult:
    current = start
    start.requires_grad_(True)
    out = model(start)
    (start_grad,) = torch.autograd.grad([out], [start], torch.ones_like(out))

    assert torch.allclose(
        start_grad,
        start_grad[0].unsqueeze(0),
        rtol=grad_rtol,
        atol=grad_atol,
    )

    scale = init_scale * current.new_ones(1, start.shape[1])
    chain = []
    accept_ratio = []
    scale_history = []
    for i in tqdm.trange(n_warmup + n_steps, disable=not show_progress):
        offset = scale * torch.randn_like(current)
        proposals = current + offset
        out = model(proposals)
        (grad,) = torch.autograd.grad([out], [proposals], torch.ones_like(out))
        # f(x') / f(x)
        accept = (
            torch.isclose(
                start_grad,
                grad,
                grad_rtol,
                grad_atol,
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

        current = proposals * accept + (1 - accept) * current
        if i >= n_warmup:
            accept_ratio.append(accept.mean())
            chain.append(current)
            scale_history.append(scale.clone())
    return SamplingResult(
        torch.stack(chain),
        torch.stack(accept_ratio),
        torch.cat(scale_history),
        start,
        start_grad,
        init_scale,
        n_steps,
        grad_rtol,
        grad_atol,
        n_warmup,
    )


# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SampleGridArgs(savethat.Args):
    n_grid_points: int = 500
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class SampleGridResult(savethat.Args):
    grid: torch.Tensor
    logits: torch.Tensor
    grads_logit_0: torch.Tensor
    grads_logit_1: torch.Tensor


class SampleGrid(savethat.Node[SampleGridArgs, SampleGridResult]):
    """Uniform sampling of the input space in [-1, 1]."""

    def sample_grid(self) -> torch.Tensor:
        grid_line = torch.linspace(-1, 1, self.args.n_grid_points)
        grid = torch.meshgrid(grid_line, grid_line, indexing="xy")
        x_a = grid[0].flatten()
        x_b = grid[1].flatten()
        return torch.stack([x_a, x_b], dim=1)

    def construct_model(self) -> nn.Module:
        torch.manual_seed(2)

        net = dtd.MLP(
            n_layers=3,
            input_size=2,
            hidden_size=30,
            output_size=2,
        )

        def weight_scale(m: nn.Module) -> None:
            for p in m.parameters():
                p.data[p.data > 0] = 1.2 * p.data[p.data > 0]
            if isinstance(m, dtd.LinearReLU):
                m.linear.bias.data = -m.linear.bias.data.abs()

        net.apply(weight_scale)
        return net

    def _run(self):
        torch.manual_seed(self.args.seed)
        net = self.construct_model()

        with open(self.output_dir / "model.ckpt", "wb") as f:
            torch.save(net.state_dict(), f)
        grid = self.sample_grid()
        grid.requires_grad_(True)

        logits = net(grid)

        (grads_logit_0,) = torch.autograd.grad(
            logits[:, 0],
            grid,
            grad_outputs=torch.ones_like(logits[:, 0]),
            retain_graph=True,
        )
        (grads_logit_1,) = torch.autograd.grad(
            logits[:, 1],
            grid,
            grad_outputs=torch.ones_like(logits[:, 0]),
        )

        return SampleGridResult(
            grid,
            logits,
            grads_logit_0,
            grads_logit_1,
        )
