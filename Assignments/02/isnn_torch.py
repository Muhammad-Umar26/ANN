from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, positive: bool):
        super().__init__()
        self.positive = positive
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight_raw)

    def weight(self) -> torch.Tensor:
        if self.positive:
            return F.softplus(self.weight_raw)
        return self.weight_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight().t() + self.bias


@dataclass
class TorchISNNConfig:
    x_layers: int
    x_width: int
    y_layers: int
    y_width: int
    z_layers: int
    z_width: int
    t_layers: int
    t_width: int


class ISNN1Torch(nn.Module):
    def __init__(self, cfg: TorchISNNConfig):
        super().__init__()
        self.cfg = cfg

        self.y_layers = nn.ModuleList()
        prev = 1
        for _ in range(cfg.y_layers):
            self.y_layers.append(ConstrainedLinear(prev, cfg.y_width, positive=True))
            prev = cfg.y_width

        self.z_layers = nn.ModuleList()
        prev = 1
        for _ in range(cfg.z_layers):
            self.z_layers.append(ConstrainedLinear(prev, cfg.z_width, positive=False))
            prev = cfg.z_width

        self.t_layers = nn.ModuleList()
        prev = 1
        for _ in range(cfg.t_layers):
            self.t_layers.append(ConstrainedLinear(prev, cfg.t_width, positive=True))
            prev = cfg.t_width

        self.x0_to_x = ConstrainedLinear(1, cfg.x_width, positive=False)
        self.y_to_x = ConstrainedLinear(cfg.y_width, cfg.x_width, positive=True)
        self.z_to_x = ConstrainedLinear(cfg.z_width, cfg.x_width, positive=False)
        self.t_to_x = ConstrainedLinear(cfg.t_width, cfg.x_width, positive=True)

        self.x_layers = nn.ModuleList()
        for _ in range(max(0, cfg.x_layers - 1)):
            self.x_layers.append(ConstrainedLinear(cfg.x_width, cfg.x_width, positive=True))

        self.out = nn.Linear(cfg.x_width, 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _sigma_mc(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def _sigma_m(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def _sigma_a(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def forward(self, x_all: torch.Tensor) -> torch.Tensor:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        y = y0
        for layer in self.y_layers:
            y = self._sigma_mc(layer(y))

        z = z0
        for layer in self.z_layers:
            z = self._sigma_a(layer(z))

        t = t0
        for layer in self.t_layers:
            t = self._sigma_m(layer(t))

        x = self._sigma_mc(self.x0_to_x(x0) + self.y_to_x(y) + self.z_to_x(z) + self.t_to_x(t))
        for layer in self.x_layers:
            x = self._sigma_mc(layer(x))

        return self.out(x)


class ISNN2Torch(nn.Module):
    def __init__(self, cfg: TorchISNNConfig):
        super().__init__()
        self.cfg = cfg

        self.y_layers = nn.ModuleList()
        prev = 1
        for _ in range(cfg.y_layers):
            self.y_layers.append(ConstrainedLinear(prev, cfg.y_width, positive=True))
            prev = cfg.y_width

        self.z_layers = nn.ModuleList()
        prev = 1
        for _ in range(cfg.z_layers):
            self.z_layers.append(ConstrainedLinear(prev, cfg.z_width, positive=False))
            prev = cfg.z_width

        self.t_layers = nn.ModuleList()
        prev = 1
        for _ in range(cfg.t_layers):
            self.t_layers.append(ConstrainedLinear(prev, cfg.t_width, positive=True))
            prev = cfg.t_width

        self.x0_to_x_first = ConstrainedLinear(1, cfg.x_width, positive=False)
        self.y0_to_x_first = ConstrainedLinear(1, cfg.x_width, positive=True)
        self.z0_to_x_first = ConstrainedLinear(1, cfg.x_width, positive=False)
        self.t0_to_x_first = ConstrainedLinear(1, cfg.x_width, positive=True)

        self.x_layers = nn.ModuleList()
        self.x0_skip = nn.ModuleList()
        self.xy = nn.ModuleList()
        self.xz = nn.ModuleList()
        self.xt = nn.ModuleList()

        for _ in range(max(0, cfg.x_layers - 1)):
            self.x_layers.append(ConstrainedLinear(cfg.x_width, cfg.x_width, positive=True))
            self.x0_skip.append(ConstrainedLinear(1, cfg.x_width, positive=False))
            self.xy.append(ConstrainedLinear(cfg.y_width, cfg.x_width, positive=True))
            self.xz.append(ConstrainedLinear(cfg.z_width, cfg.x_width, positive=False))
            self.xt.append(ConstrainedLinear(cfg.t_width, cfg.x_width, positive=True))

        self.out = nn.Linear(cfg.x_width, 1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _sigma_mc(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def _sigma_m(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def _sigma_a(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def forward(self, x_all: torch.Tensor) -> torch.Tensor:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        y = y0
        for layer in self.y_layers:
            y = self._sigma_mc(layer(y))

        z = z0
        for layer in self.z_layers:
            z = self._sigma_a(layer(z))

        t = t0
        for layer in self.t_layers:
            t = self._sigma_m(layer(t))

        x = self._sigma_mc(
            self.x0_to_x_first(x0)
            + self.y0_to_x_first(y0)
            + self.z0_to_x_first(z0)
            + self.t0_to_x_first(t0)
        )

        for i in range(len(self.x_layers)):
            x = self._sigma_mc(
                self.x_layers[i](x)
                + self.x0_skip[i](x0)
                + self.xy[i](y)
                + self.xz[i](z)
                + self.xt[i](t)
            )

        return self.out(x)


def default_torch_configs() -> dict:
    return {
        "isnn1": TorchISNNConfig(
            x_layers=2,
            x_width=10,
            y_layers=2,
            y_width=10,
            z_layers=2,
            z_width=10,
            t_layers=2,
            t_width=10,
        ),
        "isnn2": TorchISNNConfig(
            x_layers=2,
            x_width=15,
            y_layers=1,
            y_width=15,
            z_layers=1,
            z_width=15,
            t_layers=1,
            t_width=15,
        ),
    }
