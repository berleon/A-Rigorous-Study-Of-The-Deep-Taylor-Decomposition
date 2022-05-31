from bisect import bisect_right
from typing import Any

from torch.optim import lr_scheduler


class SequentialLR(lr_scheduler.SequentialLR):
    def __init__(
        self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False
    ):
        super().__init__(optimizer, schedulers, milestones, last_epoch, verbose)
        self._last_lr = schedulers[0].get_last_lr()
        self.last_epoch: int
        self._milestones: Any
        self._schedulers: Any

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()
        self._last_lr = self._schedulers[idx].get_last_lr()
