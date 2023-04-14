import itertools
import random
from collections import OrderedDict
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.util.parameter_initializer import Initializer

from ..util.common import get_batch_spacing, rand_color


class ReluMLP(torch.nn.Module):
    def __init__(
        self,
        coordinates: np.array,
        distances: np.array,
        initialization: np.array,
        units: int,  # mimic of number of max-affine functions
        initializer: Initializer,
        m: int,  # some better name here -> number of generated training datapoints
    ):
        super().__init__()

        self.initializer: Initializer = initializer
        self.m: int = m

        # training x/y
        self.training_coords: torch.nn.Parameter = (
            torch.nn.Parameter(torch.from_numpy(coordinates), requires_grad=False)
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

        # training d
        self.training_distances: torch.nn.Parameter = (
            torch.nn.Parameter(torch.from_numpy(distances), requires_grad=False)
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

        # evaluation x/y
        self.evaluation: torch.nn.Parameter = (
            torch.nn.Parameter(torch.randn(len(initialization), requires_grad=True))
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

        self.layers: list[torch.nn.Linear] = [
            torch.nn.Linear(self.training_coords.shape[1], units).to(
                torch.device("cuda:0")
            ),
            torch.nn.Linear(units, 1, bias=False).to(torch.device("cuda:0")),
        ]

        self.params = [param for layer in self.layers for param in list(layer)]
        self.optimizer = torch.optim.Adam(self.params, lr=0.06)

        self.activation_fn: Callable = torch.nn.ReLU()
        self.loss_fn: torch.nn.MSELoss = torch.nn.MSELoss(reduction="mean")
        self.loss: torch.Tensor

        self.initialization()

    def __call__(self, x: torch.Tensor) -> Any:
        if x is None:
            return self.forward()
        else:
            return self.eval(x)

    def initialization(self) -> None:
        a, _ = self.initializer(self.m)
        with torch.no_grad():
            self.evaluation.data = a

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[-1](self.activation_fn(self.layers[0](x)))

    def forward(self) -> None:
        self.optimizer.zero_grad()

        self.loss = self.loss_fn(
            self.layers[-1](self.activation_fn(self.layers[0](self.training_coords))),
            self.training_distances,
        )
        self.loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

    def get_loss(self) -> float:
        return self.loss.item()
