"""
Schedulers for DNNet.

Paper specifies:
  "The initial learning rate was 0.0001, which was maintained over the
   first 100 epochs and linearly decayed to zero over the next 100 epochs."

This applies to both optimisers (Adam for contrastive branch, SGD for ArcFace).
"""

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_dnnet_scheduler(
    optimizer,
    warmup_epochs: int = 5,
    total_epochs:  int = 100,
) -> LambdaLR:
    """
    Returns a LambdaLR scheduler that:
      - Keeps LR constant for epochs [0, warmup_epochs)
      - Linearly decays LR from base_lr → 0 for epochs [warmup_epochs, total_epochs)
    """
    decay_epochs = total_epochs - warmup_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / max(decay_epochs, 1)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def sync_lambda_scheduler_param_groups(scheduler: LambdaLR) -> None:
    """
    Keep LambdaLR internals aligned with optimizer param_groups.

    This is needed when new param groups are added after scheduler creation
    (e.g., unfreezing backbone mid-training).
    """
    num_opt_groups = len(scheduler.optimizer.param_groups)
    num_sched_groups = len(scheduler.base_lrs)

    if num_opt_groups <= num_sched_groups:
        return

    template_lambda = scheduler.lr_lambdas[-1]

    for idx in range(num_sched_groups, num_opt_groups):
        group_lr = scheduler.optimizer.param_groups[idx]["lr"]
        scheduler.base_lrs.append(group_lr)
        scheduler.lr_lambdas.append(template_lambda)

        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr.append(group_lr)
