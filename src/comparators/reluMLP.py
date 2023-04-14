from util.common import rand_color, get_batch_spacing
from collections import OrderedDict

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
from src.util.parameter_initializer import Initializer
import itertools
from typing import Any

class ReluMLP(torch.nn.Module):
    def __init__(
        self,
        coordinates: np.array,
        distances: np.array,
        initialization: np.array,
        units:int,                  # mimic of number of max-affine functions
        initializer: Initializer,
        m: int,                     # some better name here -> number of generated training datapoints
    ):
        super(ReluMLP, self).__init__()

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
        self.evaluation:  torch.nn.Parameter = (
            torch.nn.Parameter(torch.randn(len(initialization),requires_grad=True))
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

        self.layers:list[torch.nn.Linear] = [
            torch.nn.Linear(self.training_coords.shape[1],units).to(torch.device("cuda:0")),
            torch.nn.Linear(units,1, bias=False).to(torch.device("cuda:0"))
        ]

        self.params = sum([ list(layer) for layer in self.layers])


        self.optimizer = torch.optim.Adam(self.params)

        self.activation_fn = torch.nn.ReLU()
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        self.initialization()


    def __call__(self, x:torch.Tensor) -> Any:
        if x is None:
            return self.forward()
        else:
            return self.eval(x)


    def initialization(self) -> None:
        a,_ = self.initializer(self.m)
        with torch.no_grad():
            self.evaluation.data = a


    def eval(self,x:torch.Tensor) -> torch.Tensor:
        return self.layers[-1](self.activation_fn(self.layers[0](x)))


    def forward(self) -> None:
        self.optimizer.zero_grad()
        training_loss = self.layers[-1](self.activation_fn(self.layers[0](self.training_coords)))
        loss = self.loss_fn(training_loss,self.training_distances)
        loss.backward()
        self.optimizer.step()
