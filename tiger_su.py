import re
import torch
from typing import Union
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


def root_mean_square(x, dim=None, keepdim=False):
    """Root Mean Square"""
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=keepdim))


def piecewise_linear(t, schedule, from_zero=True):
    """Piecewise Linear Function"""
    schedule = sorted(schedule.items())
    if from_zero and schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    t = float(t)
    x = float(schedule[0][1])
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = float(schedule[i][1])
        x = x if t >= t_begin else x_begin

    return x


class TigerWithScheduler(Optimizer):
    r"""Tiger Optimizer
        A PyTorch implementation of the Tiger optimizer with a scheduler based on 
        https://github.com/bojone/tiger/blob/main/tiger.py
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        beta=0.965,
        weight_decay=0.01,
        grad_accum_steps=1,
        lr_schedule={0: 1},
        shrink_ratio=0.99,
    ):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {lr}')
        if not 0.0 <= beta < 1.0:
            raise ValueError('Invalid beta parameter: {beta}')
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, 
                        grad_accum_steps=grad_accum_steps, 
                        lr_schedule={int(i): j for i, j in lr_schedule.items()},
                        shrink_ratio=shrink_ratio)
        super().__init__(params, defaults)
        self.param_names = {id(p): name for name, p in self._get_param_names(params)}
        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self._step_count = 0
        self.step()

    def _get_param_names(self, params):
        """Helper function to get parameter names"""
        if isinstance(params, dict):
            for name, param in params.items():
                if isinstance(param, torch.Tensor):
                    yield name, param
                else:
                    for sub_name, sub_param in self._get_param_names(param):
                        yield f'{name}.{sub_name}', sub_param
        elif isinstance(params, (list, tuple)):
            for i, param in enumerate(params):
                if isinstance(param, torch.Tensor):
                    yield str(i), param
                else:
                    for sub_name, sub_param in self._get_param_names(param):
                        yield f'{i}.{sub_name}', sub_param

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            weight_decay = group['weight_decay']
            grad_accum_steps = group['grad_accum_steps']
            shrink_ratio = group['shrink_ratio']
            lr_schedule = group['lr_schedule']

            t = self._step_count
            b1 = beta if (t % grad_accum_steps == 0) else 1.0
            b2 = (1 - beta) / grad_accum_steps
            lr = group['lr'] * piecewise_linear(t, lr_schedule)
            lr = lr if ((t + 1) % grad_accum_steps == 0) else 0.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Tiger does not support sparse gradients')

                state = self.state[p]
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p.data)

                m = state['m']
                is_nan = torch.isnan(grad).any()
                b1 = 1.0 if is_nan else b1
                g = torch.zeros_like(grad) if is_nan else grad

                param_name = self.param_names.get(id(p), "")

                if re.findall('bias|beta|gamma', param_name):
                    lr *= 0.5
                    weight_decay = 0.0
                elif 'embeddings' in param_name:
                    lr *= root_mean_square(p.data, dim=-1, keepdim=True)
                else:
                    lr *= root_mean_square(p.data)

                m.mul_(b1).add_(b2 * g)
                update = (torch.sign(m) + weight_decay * p.data) * lr
                p.data.add_(-update)

                if is_nan:
                    p.data.mul_(shrink_ratio).add_((1 - shrink_ratio) * torch.sign(m))

        self._step_count += 1
        
        return loss
