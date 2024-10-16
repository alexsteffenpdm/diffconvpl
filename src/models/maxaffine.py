from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from ..util.common import get_batch_spacing
from ..util.parameter_initializer import Initializer


class MultiDimMaxAffineFunction(torch.nn.Module):
    def __init__(
        self,
        m: int,
        k: int,
        dim: int,
        x: np.array,
        y: torch.tensor,
        signs: np.array,
        initializer: Initializer,
        batchsize: int = 2**32,
    ):
        super().__init__()

        self.initializer: Initializer = initializer

        self.x: torch.nn.Parameter = (
            torch.nn.Parameter(torch.from_numpy(x), requires_grad=False)
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

        self.y: torch.Tensor = y.type(torch.FloatTensor).to(torch.device("cuda:0"))

        self.s: torch.nn.Parameter = torch.nn.Parameter(
            torch.from_numpy(signs), requires_grad=False
        )

        self.a: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand((k, m, dim), dtype=torch.float32), requires_grad=True
        )

        self.b: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand((k, m), dtype=torch.float32), requires_grad=True
        )
        self.k: int = k
        self.dim: int = dim
        self.m: int = m

        self.func: Callable = lambda t: torch.max(t, dim=-1)[0]
        self.optimizer = torch.optim.Adam([self.a, self.b], lr=0.06)

        self.activation_fn: Callable = lambda t: torch.max(t, dim=-1)[0]
        self.loss_fn: torch.nn.MSELoss = torch.nn.MSELoss(reduction="mean")
        self.loss: torch.Tensor

        self.batch_size: int
        if batchsize == 2**32:
            self.batch_size = self.x.shape[0]
        else:
            self.batch_size = batchsize

        self.batches: list[tuple[int, int]] = []
        self.initialization()

    def __call__(self, batching: bool) -> torch.Tensor:
        return self.forward(batching)

    def initialization(self) -> None:
        a, b = self.initializer(self.m)
        with torch.no_grad():
            for ki in range(self.k):
                self.a[ki].data = a
                self.b[ki].data = b
        assert self.a.shape == (self.k, self.m, self.dim)

    def eval(self, ki, x=None) -> torch.Tensor:
        if len(self.x.shape) == 1:
            self.x = self.x[:, None]

        if x == None:
            return self.func(self.x @ self.a[ki].T + self.b[ki])
        else:
            return self.func(x @ self.a[ki].T + self.b[ki])

    def batch_eval(self, s, e) -> torch.Tensor:
        if len(self.x.shape) == 1:
            self.x = self.x[:, None]

        U: torch.Tensor = (
            torch.einsum("bn, kmn -> bkm", self.x[s:e], self.a) + self.b[None]
        )
        V: torch.Tensor = self.func(U)
        W: torch.Tensor = (self.s[None] * V).sum(dim=-1, keepdim=True)
        return W

    def forward(self, batching: bool) -> None:
        self.optimizer.zero_grad()
        if batching:
            self.loss = self.loss_fn(
                torch.cat([self.batch_eval(s=b[0], e=b[1]) for b in self.batches]),
                self.y,
            )
        else:
            self.loss = self.loss_fn(
                torch.stack(
                    [
                        si * self.eval(ki=ki, x=None)
                        for ki, si in zip(range(self.k), self.s)
                    ]
                ).sum(dim=0),
                self.y,
            )

        self.loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

    def bench(self, y_target, granularity=0.005) -> None:
        if len(self.x.shape) == 1:
            self.x = self.x[:, None]

        for xi in tqdm(np.linspace(0.0, 1.0, int(1.0 / granularity))[1:]):
            try:
                entries = int(self.x.shape[0] * xi)
                l = (self.batch_eval(s=0, e=entries) - y_target).pow(2).mean()
            except RuntimeError:
                break
            else:
                self.batch_size = entries
        self.batches = get_batch_spacing(self.batch_size, self.x.shape[0])

    def generate_sdf_plot_data(
        self, x: np.ndarray, y: np.ndarray, k: int
    ) -> np.ndarray:
        x_flat: torch.Tensor = (
            torch.from_numpy(x.flatten())
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )
        y_flat: torch.Tensor = (
            torch.from_numpy(y.flatten())
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )
        z: torch.Tensor = torch.zeros_like(x_flat)
        z.data = self.eval(ki=k, x=torch.vstack([x_flat, y_flat]).T) * self.s[k]

        return z.cpu().detach().numpy().reshape(len(x), len(y))

    # def error_propagation(
    #     self, spacing: float, min: float, max: float
    # ) -> Tuple[np.array, np.array]:
    #     def model_output(data: torch.FloatTensor, k: int, s: torch.FloatTensor):
    #         prediction: torch.Tensor = torch.zeros((k), dtype=torch.float32).to(
    #             torch.device("cuda:0")
    #         )
    #         for ki, si in zip(range(k), s):
    #             prediction[ki] = si * self.eval(ki=ki, x=data)
    #         return prediction.sum()

    #     print(
    #         f"\tCollecting model error propagation (from {min} to {max} with granularity: {spacing})"
    #     )

    #     points: int = int((abs(min) + abs(max)) / spacing) + 1
    #     domain: np.array = np.linspace(min, max, points)

    #     x_y: torch.Tensor = (
    #         torch.from_numpy(domain.flatten())
    #         .type(torch.FloatTensor)
    #         .to(torch.device("cuda:0"))
    #     )
    #     model_z: torch.Tensor = (
    #         torch.zeros_like(x_y).type(torch.FloatTensor).to(torch.device("cuda:0"))
    #     )
    #     target_z: torch.Tensor = (
    #         torch.zeros_like(x_y).type(torch.FloatTensor).to(torch.device("cuda:0"))
    #     )

    #     for i in tqdm(range(len(x_y))):
    #         model_z[i].data = model_output(
    #             data=torch.stack([x_y[i], x_y[i]]), k=self.k, s=self.s
    #         )
    #         target_z[i].data = self.target.as_lambda("torch")(x_y[i], x_y[i])

    #     model_z: np.array = model_z.cpu().detach().numpy()
    #     target_z: np.array = target_z.cpu().detach().numpy()

    #     return domain, np.subtract(model_z, target_z)

    def get_loss(self) -> float:
        return self.loss.item()
