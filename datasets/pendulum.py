import numpy as np
import math, shelve
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
import torch
import torchdiffeq
from collections import OrderedDict

MAX = np.iinfo(np.int32).max

class DampledPendulum(Dataset):
    __default_params = OrderedDict(omega0_square=(2 * math.pi / 12) ** 2, alpha=0.2)

    def __init__(self, path, num_seq, time_horizon, dt, params=None, group='train'):
        super().__init__()
        self.len = num_seq
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)
        self.group = group
        self.data = shelve.open(path)

    def _f(self, t, x):  # coords = [q,p]
        omega0_square, alpha = list(self.params.values())

        q, p = np.split(x, 2)
        dqdt = p
        dpdt = -omega0_square * np.sin(q) - alpha * p
        return np.concatenate([dqdt, dpdt], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else MAX-seed)
        y0 = np.random.rand(2) * 2.0 - 1
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        return y0

    def __getitem__(self, index):
        t_eval = np.arange(0, self.time_horizon, self.dt)
        if self.data.get(str(index)) is None:
            y0 = self._get_initial_condition(index)
            states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), y0=y0, method='DOP853', t_eval=t_eval, rtol=1e-10).y
            self.data[str(index)] = states
            states = torch.from_numpy(states).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()
        return {'states': states, 't': t_eval.float()}

    def __len__(self):
        return self.len