import torch
import numpy as np
from typing import Tuple, Callable
from tqdm import tqdm

from .util.common import get_batch_spacing
from .util.parameter_initializer import Initializer


class MultiDimMaxAffineFunction(torch.nn.Module):
    def __init__(
        self,
        m: int,
        k: int,
        dim: int,
        x: np.array,
        signs: np.array,
        # target: Target,
        initializer: Initializer,
        batchsize: int = 2**32,
    ):
        super(MultiDimMaxAffineFunction, self).__init__()

        self.initializer: Initializer = initializer
        # self.target: Target = target

        self.x: torch.nn.Parameter = (
            torch.nn.Parameter(torch.from_numpy(x), requires_grad=False)
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

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
        # self.func: Callable = lambda t: torch.logsumexp(torch.relu(t), dim=-1)
        self.gradfunc: Callable = lambda t: torch.argmax(t, dim=-1)
        # self.func:Callable = lambda t: torch.logsumexp(t, dim=-1)
        if batchsize == 2**32:
            self.batch_size: int = self.x.shape[0]
        else:
            self.batch_size: int = batchsize

        self.batches: list[int] = []
        self.initialization()

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

    def forward(self, batching: bool) -> torch.Tensor:
        if batching:
            return torch.cat([self.batch_eval(s=b[0], e=b[1]) for b in self.batches])
        else:
            return torch.stack(
                [si * self.eval(ki=ki, x=None) for ki, si in zip(range(self.k), self.s)]
            ).sum(dim=0)

    def batch_diff(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        diff: torch.Tensor = torch.zeros(1, dtype=torch.float32, requires_grad=True).to(
            torch.device("cuda:0")
        )
        for b in self.batches:
            diff += (prediction[b[0] : b[1]] - target[b[0] : b[1]]).pow(2).mean()

        return diff.mean()

    def __call__(self, batching: bool) -> torch.Tensor:
        return self.forward(batching)

    def generate_sdf_plot_data(self, spacing: float, min: float, max: float):
        def model_output(
            data: torch.FloatTensor, k: int, s: torch.FloatTensor
        ) -> torch.FloatTensor:
            prediction: torch.Tensor = torch.zeros((k), dtype=torch.float32).to(
                torch.device("cuda:0")
            )
            for ki, si in zip(range(k), s):
                prediction[ki] = self.eval(ki=ki, x=data) * si
            return prediction.sum()

        print(f"\tCollecting model evaluation (granularity: {spacing})")

        points: np.array = int((abs(min) + abs(max)) / spacing) + 1
        domain: np.array = np.linspace(min, max, points)
        x, y = np.meshgrid(domain, domain)

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

        for i in tqdm(range(len(x_flat))):
            z.data[i] = model_output(
                data=torch.stack([x_flat[i], y_flat[i]]), k=self.k, s=self.s
            )

        try:
            return x, y, z.cpu().detach().numpy().reshape(len(x), len(y))
        except:
            print("could not return")
            exit()

    def generate_sdf_plot_data_single_maxaffine_function(
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

        for i in range(len(x_flat)):
            z.data[i] = (
                self.eval(ki=k, x=torch.stack([x_flat[i], y_flat[i]])) * self.s[k]
            )

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

    def tensor_devices(self) -> None:
        tensors = {"a": self.a, "b": self.b, "s": self.s, "x": self.x}
        for k, v in tensors.items():
            print(f"Tensor {k} on device: {v.device}")

    def gradient(self, min: float, max: float):
        domain: np.array = np.linspace(min, max, self.m)
        x, y = np.meshgrid(domain, domain)
        print(np.vstack((x, y)).shape, np.vstack((x, y)))
        # domain:np.array = torch.from_numpy(np.linspace(min,max,points)).type(torch.FloatTensor).to(torch.device("cuda:0"))
        domain = (
            torch.from_numpy(np.vstack((x, y)))
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

        q = np.zeros([self.m, self.k])
        for ki in range(self.k):
            q[ki] = self.gradfunc(domain @ self.a[ki].T + self.b[ki])
            print(q[ki])
