from util.common import rand_color, get_batch_spacing
from util.target import Target

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm

class ReluMLP(torch.nn.Module):
    def __init__(self,):
        super(ReluMLP,self).__init__()
        self.layer = torch.nn.Linear(1,1) # fully connected layer
        self.func = torch.nn.ReLU()

    def __call__(self,x):
        return self.forward(x)


    def forward(self,x):
        output = self.layer(x)
        output = self.func(x)
        return output
    