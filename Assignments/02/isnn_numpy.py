from dataclasses import dataclass

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid to avoid overflow in exp for large |x|.
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


@dataclass
class NumpyISNNConfig:
    x_layers: int
    x_width: int
    y_layers: int
    y_width: int
    z_layers: int
    z_width: int
    t_layers: int
    t_width: int


class ConstrainedLinearNumpy:
    def __init__(self, in_features: int, out_features: int, positive: bool, rng: np.random.Generator):
        self.positive = positive
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight_raw = rng.normal(0.0, scale, size=(out_features, in_features))
        self.bias = np.zeros((1, out_features), dtype=np.float64)

        self.grad_weight_raw = np.zeros_like(self.weight_raw)
        self.grad_bias = np.zeros_like(self.bias)

        self.last_x = None
        self.last_weight = None

    def _weight(self) -> np.ndarray:
        if self.positive:
            return softplus(self.weight_raw)
        return self.weight_raw

    def forward(self, x: np.ndarray) -> np.ndarray:
        w = self._weight()
        self.last_x = x
        self.last_weight = w
        return x @ w.T + self.bias

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_w = grad_out.T @ self.last_x
        self.grad_bias = np.sum(grad_out, axis=0, keepdims=True)

        if self.positive:
            self.grad_weight_raw = grad_w * sigmoid(self.weight_raw)
        else:
            self.grad_weight_raw = grad_w

        # Keep updates bounded in the manual implementation.
        np.clip(self.grad_weight_raw, -10.0, 10.0, out=self.grad_weight_raw)
        np.clip(self.grad_bias, -10.0, 10.0, out=self.grad_bias)

        grad_x = grad_out @ self.last_weight
        return grad_x

    def step(self, lr: float) -> None:
        self.weight_raw -= lr * self.grad_weight_raw
        self.bias -= lr * self.grad_bias


class ISNN1Numpy:
    def __init__(self, cfg: NumpyISNNConfig, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)

        self.y_layers = []
        prev = 1
        for _ in range(cfg.y_layers):
            self.y_layers.append(ConstrainedLinearNumpy(prev, cfg.y_width, positive=True, rng=rng))
            prev = cfg.y_width

        self.z_layers = []
        prev = 1
        for _ in range(cfg.z_layers):
            self.z_layers.append(ConstrainedLinearNumpy(prev, cfg.z_width, positive=False, rng=rng))
            prev = cfg.z_width

        self.t_layers = []
        prev = 1
        for _ in range(cfg.t_layers):
            self.t_layers.append(ConstrainedLinearNumpy(prev, cfg.t_width, positive=True, rng=rng))
            prev = cfg.t_width

        self.x0_to_x = ConstrainedLinearNumpy(1, cfg.x_width, positive=False, rng=rng)
        self.y_to_x = ConstrainedLinearNumpy(cfg.y_width, cfg.x_width, positive=True, rng=rng)
        self.z_to_x = ConstrainedLinearNumpy(cfg.z_width, cfg.x_width, positive=False, rng=rng)
        self.t_to_x = ConstrainedLinearNumpy(cfg.t_width, cfg.x_width, positive=True, rng=rng)

        self.x_layers = []
        for _ in range(max(0, cfg.x_layers - 1)):
            self.x_layers.append(ConstrainedLinearNumpy(cfg.x_width, cfg.x_width, positive=True, rng=rng))

        self.out = ConstrainedLinearNumpy(cfg.x_width, 1, positive=False, rng=rng)

        self.cache = {}

    def forward(self, x_all: np.ndarray) -> np.ndarray:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        y = y0
        y_pre = []
        y_post = []
        for layer in self.y_layers:
            pre = layer.forward(y)
            y = softplus(pre)
            y_pre.append(pre)
            y_post.append(y)

        z = z0
        z_pre = []
        z_post = []
        for layer in self.z_layers:
            pre = layer.forward(z)
            z = np.tanh(pre)
            z_pre.append(pre)
            z_post.append(z)

        t = t0
        t_pre = []
        t_post = []
        for layer in self.t_layers:
            pre = layer.forward(t)
            t = softplus(pre)
            t_pre.append(pre)
            t_post.append(t)

        x_pre = self.x0_to_x.forward(x0) + self.y_to_x.forward(y) + self.z_to_x.forward(z) + self.t_to_x.forward(t)
        x = softplus(x_pre)

        x_h_pre = []
        x_h_post = []
        for layer in self.x_layers:
            pre = layer.forward(x)
            x = softplus(pre)
            x_h_pre.append(pre)
            x_h_post.append(x)

        y_hat = self.out.forward(x)

        self.cache = {
            "x0": x0,
            "y_pre": y_pre,
            "y_post": y_post,
            "z_pre": z_pre,
            "z_post": z_post,
            "t_pre": t_pre,
            "t_post": t_post,
            "x_pre": x_pre,
            "x_after_first": softplus(x_pre),
            "x_h_pre": x_h_pre,
            "x_h_post": x_h_post,
        }
        return y_hat

    def backward(self, grad_y_hat: np.ndarray) -> None:
        grad_x = self.out.backward(grad_y_hat)

        for i in range(len(self.x_layers) - 1, -1, -1):
            pre = self.cache["x_h_pre"][i]
            grad_pre = grad_x * sigmoid(pre)
            grad_x = self.x_layers[i].backward(grad_pre)

        grad_x_pre = grad_x * sigmoid(self.cache["x_pre"])
        grad_x0 = self.x0_to_x.backward(grad_x_pre)
        grad_y_top = self.y_to_x.backward(grad_x_pre)
        grad_z_top = self.z_to_x.backward(grad_x_pre)
        grad_t_top = self.t_to_x.backward(grad_x_pre)

        for i in range(len(self.y_layers) - 1, -1, -1):
            pre = self.cache["y_pre"][i]
            grad_pre = grad_y_top * sigmoid(pre)
            grad_y_top = self.y_layers[i].backward(grad_pre)

        for i in range(len(self.z_layers) - 1, -1, -1):
            pre = self.cache["z_pre"][i]
            z_post = self.cache["z_post"][i]
            grad_pre = grad_z_top * (1.0 - z_post * z_post)
            grad_z_top = self.z_layers[i].backward(grad_pre)

        for i in range(len(self.t_layers) - 1, -1, -1):
            pre = self.cache["t_pre"][i]
            grad_pre = grad_t_top * sigmoid(pre)
            grad_t_top = self.t_layers[i].backward(grad_pre)

        _ = grad_x0, grad_y_top, grad_z_top, grad_t_top

    def step(self, lr: float) -> None:
        for layer in self.y_layers + self.z_layers + self.t_layers:
            layer.step(lr)
        self.x0_to_x.step(lr)
        self.y_to_x.step(lr)
        self.z_to_x.step(lr)
        self.t_to_x.step(lr)
        for layer in self.x_layers:
            layer.step(lr)
        self.out.step(lr)


class ISNN2Numpy:
    def __init__(self, cfg: NumpyISNNConfig, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)

        self.y_layers = []
        prev = 1
        for _ in range(cfg.y_layers):
            self.y_layers.append(ConstrainedLinearNumpy(prev, cfg.y_width, positive=True, rng=rng))
            prev = cfg.y_width

        self.z_layers = []
        prev = 1
        for _ in range(cfg.z_layers):
            self.z_layers.append(ConstrainedLinearNumpy(prev, cfg.z_width, positive=False, rng=rng))
            prev = cfg.z_width

        self.t_layers = []
        prev = 1
        for _ in range(cfg.t_layers):
            self.t_layers.append(ConstrainedLinearNumpy(prev, cfg.t_width, positive=True, rng=rng))
            prev = cfg.t_width

        self.x0_to_x_first = ConstrainedLinearNumpy(1, cfg.x_width, positive=False, rng=rng)
        self.y0_to_x_first = ConstrainedLinearNumpy(1, cfg.x_width, positive=True, rng=rng)
        self.z0_to_x_first = ConstrainedLinearNumpy(1, cfg.x_width, positive=False, rng=rng)
        self.t0_to_x_first = ConstrainedLinearNumpy(1, cfg.x_width, positive=True, rng=rng)

        self.x_layers = []
        self.x0_skip = []
        self.xy = []
        self.xz = []
        self.xt = []
        for _ in range(max(0, cfg.x_layers - 1)):
            self.x_layers.append(ConstrainedLinearNumpy(cfg.x_width, cfg.x_width, positive=True, rng=rng))
            self.x0_skip.append(ConstrainedLinearNumpy(1, cfg.x_width, positive=False, rng=rng))
            self.xy.append(ConstrainedLinearNumpy(cfg.y_width, cfg.x_width, positive=True, rng=rng))
            self.xz.append(ConstrainedLinearNumpy(cfg.z_width, cfg.x_width, positive=False, rng=rng))
            self.xt.append(ConstrainedLinearNumpy(cfg.t_width, cfg.x_width, positive=True, rng=rng))

        self.out = ConstrainedLinearNumpy(cfg.x_width, 1, positive=False, rng=rng)
        self.cache = {}

    def forward(self, x_all: np.ndarray) -> np.ndarray:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        y = y0
        y_pre = []
        y_post = []
        for layer in self.y_layers:
            pre = layer.forward(y)
            y = softplus(pre)
            y_pre.append(pre)
            y_post.append(y)

        z = z0
        z_pre = []
        z_post = []
        for layer in self.z_layers:
            pre = layer.forward(z)
            z = np.tanh(pre)
            z_pre.append(pre)
            z_post.append(z)

        t = t0
        t_pre = []
        t_post = []
        for layer in self.t_layers:
            pre = layer.forward(t)
            t = softplus(pre)
            t_pre.append(pre)
            t_post.append(t)

        x_first_pre = (
            self.x0_to_x_first.forward(x0)
            + self.y0_to_x_first.forward(y0)
            + self.z0_to_x_first.forward(z0)
            + self.t0_to_x_first.forward(t0)
        )
        x = softplus(x_first_pre)

        x_h_pre = []
        for i in range(len(self.x_layers)):
            pre = (
                self.x_layers[i].forward(x)
                + self.x0_skip[i].forward(x0)
                + self.xy[i].forward(y)
                + self.xz[i].forward(z)
                + self.xt[i].forward(t)
            )
            x = softplus(pre)
            x_h_pre.append(pre)

        y_hat = self.out.forward(x)

        self.cache = {
            "x0": x0,
            "y_pre": y_pre,
            "y_post": y_post,
            "z_pre": z_pre,
            "z_post": z_post,
            "t_pre": t_pre,
            "t_post": t_post,
            "x_first_pre": x_first_pre,
            "x_h_pre": x_h_pre,
        }

        return y_hat

    def backward(self, grad_y_hat: np.ndarray) -> None:
        grad_x = self.out.backward(grad_y_hat)

        grad_y = np.zeros((grad_x.shape[0], self.cfg.y_width))
        grad_z = np.zeros((grad_x.shape[0], self.cfg.z_width))
        grad_t = np.zeros((grad_x.shape[0], self.cfg.t_width))

        for i in range(len(self.x_layers) - 1, -1, -1):
            pre = self.cache["x_h_pre"][i]
            grad_pre = grad_x * sigmoid(pre)
            grad_x = self.x_layers[i].backward(grad_pre)
            _ = self.x0_skip[i].backward(grad_pre)
            grad_y += self.xy[i].backward(grad_pre)
            grad_z += self.xz[i].backward(grad_pre)
            grad_t += self.xt[i].backward(grad_pre)

        grad_first_pre = grad_x * sigmoid(self.cache["x_first_pre"])
        _ = self.x0_to_x_first.backward(grad_first_pre)
        _ = self.y0_to_x_first.backward(grad_first_pre)
        _ = self.z0_to_x_first.backward(grad_first_pre)
        _ = self.t0_to_x_first.backward(grad_first_pre)

        for i in range(len(self.y_layers) - 1, -1, -1):
            pre = self.cache["y_pre"][i]
            grad_pre = grad_y * sigmoid(pre)
            grad_y = self.y_layers[i].backward(grad_pre)

        for i in range(len(self.z_layers) - 1, -1, -1):
            z_post = self.cache["z_post"][i]
            grad_pre = grad_z * (1.0 - z_post * z_post)
            grad_z = self.z_layers[i].backward(grad_pre)

        for i in range(len(self.t_layers) - 1, -1, -1):
            pre = self.cache["t_pre"][i]
            grad_pre = grad_t * sigmoid(pre)
            grad_t = self.t_layers[i].backward(grad_pre)

    def step(self, lr: float) -> None:
        for layer in self.y_layers + self.z_layers + self.t_layers:
            layer.step(lr)

        self.x0_to_x_first.step(lr)
        self.y0_to_x_first.step(lr)
        self.z0_to_x_first.step(lr)
        self.t0_to_x_first.step(lr)

        for i in range(len(self.x_layers)):
            self.x_layers[i].step(lr)
            self.x0_skip[i].step(lr)
            self.xy[i].step(lr)
            self.xz[i].step(lr)
            self.xt[i].step(lr)

        self.out.step(lr)


def default_numpy_configs() -> dict:
    return {
        "isnn1": NumpyISNNConfig(
            x_layers=2,
            x_width=10,
            y_layers=2,
            y_width=10,
            z_layers=2,
            z_width=10,
            t_layers=2,
            t_width=10,
        ),
        "isnn2": NumpyISNNConfig(
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
