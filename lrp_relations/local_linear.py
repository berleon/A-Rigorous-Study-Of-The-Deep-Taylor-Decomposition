import torch
from torch import nn


def sample(
    model: nn.Module,
    start: torch.Tensor,
    init_scale: float = 1e-6,
    n_steps: int = 1000,
    grad_rtol: float = 1e-3,
    grad_atol: float = 1e-5,
    n_warmup: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    for i in range(n_warmup + n_steps):
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
        if accept.mean() > 0.8:
            scale = 1.10 * scale
        elif accept.mean() < 0.5:
            scale = 0.80 * scale

        current = proposals * accept + (1 - accept) * proposals
        if i >= n_warmup:
            accept_ratio.append(accept.mean())
            chain.append(current)
    return torch.stack(chain), torch.stack(accept_ratio)


@dataclasses.dataclass(frozen=True)
class SampleInputsUnifromArgs(savethat.Args):
    n_grid_points: int = 500


@dataclasses.dataclass(frozen=True)
class SampleInputsUnifromResult(savethat.Args):
    grid: torch.Tensor
    grads_logit_0: torch.Tensor
    grads_logit_1: torch.Tensor


def construct_model(args: SampleInputsUnifromArgs) -> nn.Module:
    torch.manual_seed(2)

    net = dtd.NLayerMLP(
        n_layers=3,
        input_size=2,
        hidden_size=30,
        output_size=2,
    )

    def weight_scale(m: nn.Module) -> nn.Module:
        for p in m.parameters():
            p.data[p.data > 0] = 1.2 * p.data[p.data > 0]
        if isinstance(m, dtd.LinearReLU):
            m.linear.bias.data = -m.linear.bias.data.abs()
        return m

    net.apply(weight_scale)
    return net

