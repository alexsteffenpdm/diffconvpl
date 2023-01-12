from .util.common import rand_color, get_batch_spacing
from .util.target import Target

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm


class MultiDimMaxAffineFunction(torch.nn.Module):
    def __init__(
        self,
        m: int,
        k: int,
        dim: int,
        x,
        signs: List[float],
        target: Target,
        batchsize: int = 2**32,
    ):
        super().__init__()

        self.target = target

        self.x = (
            torch.nn.Parameter(torch.from_numpy(x), requires_grad=False)
            .type(torch.FloatTensor)
            .to(torch.device("cuda:0"))
        )

        self.s = torch.nn.Parameter(
            torch.from_numpy(signs), requires_grad=False
        )  # .to(torch.device("cuda:0"))

        self.a = torch.nn.Parameter(
            torch.rand((k, m, dim), dtype=torch.float32), requires_grad=True
        )  # .to(torch.device("cuda:0"))

        self.b = torch.nn.Parameter(
            torch.rand((k, m), dtype=torch.float32), requires_grad=True
        )  # .to(torch.device("cuda:0"))
        self.k = k
        self.dim = dim
        self.m = m
        self.domains = np.linspace(-1.0, 1.0, self.k + 1)

        # self.func = lambda t: torch.max(t,dim=-1)[0]
        self.func = lambda t: torch.logsumexp(t, dim=-1)
        if batchsize == 2**32:
            self.batch_size: int = self.x.shape[0]
        else:
            self.batch_size = batchsize

        self.batches = []
        self.param_init()



    def param_init(self):
        def _gauss_dist(s: int, e: int, entries: int):
            dist = np.asarray([random.gauss(0.1, 1.0) for _ in range(entries)])
            domain_range = np.max(dist) + (
                -1 * np.min(dist) if np.min(dist) < 0.0 else np.min(dist)
            )
            s = []
            for d in dist:
                v = 2 * abs(d) / domain_range
                if d < 0.0:
                    s.append(-v)
                else:
                    s.append(v)
            return s

        domains = np.linspace(-1.0, 1.0, self.k + 1)

        Arr_a = []
        Arr_b = []
        for ki in range(self.k):
            a_ki = []
            b_ki = []
            for xi,yi in zip(_gauss_dist(domains[ki], domains[ki + 1], entries=self.m),_gauss_dist(domains[ki], domains[ki + 1], entries=self.m)):
                pi = torch.tensor(np.array([xi,yi]),dtype=torch.float32,requires_grad=True)
                y = self.target.as_lambda("torch")(pi[0],pi[1])            
                y.backward()               
                pi.retain_grad()

                a_ki.append(np.asanyarray([pi.grad[0],pi.grad[1]]))
                b_ki.append(y.item())
            Arr_a.append(a_ki)
            Arr_b.append(b_ki)
      
        with torch.no_grad():
            for ki in range(self.k):
                for a_ki in Arr_a[ki]: 
                    self.a[ki] = torch.from_numpy(a_ki).type(torch.FloatTensor)
                self.b[ki] = torch.from_numpy(np.asanyarray(Arr_b[ki])).type(torch.FloatTensor)

            assert self.a.shape == (self.k, self.m, self.dim)

    def eval(self, ki, x=None):
        if len(self.x.shape) == 1:
            self.x = self.x[:, None]
        if x == None:
            return self.func(self.x @ self.a[ki].T + self.b[ki])
        else:
            return self.func(x @ self.a[ki].T + self.b[ki])

    def batch_eval(self, s, e, x=None):
        if len(self.x.shape) == 1:
            self.x = self.x[:, None]

        U = torch.einsum("bn, kmn -> bkm", self.x[s:e], self.a) + self.b[None]
        V = self.func(U)
        W = (self.s[None] * V).sum(dim=-1, keepdim=True)
        return W

    def bench(self, y_target, granularity=0.005):
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

    def forward(self, batching: bool):
        if batching:
            return torch.cat([self.batch_eval(s=b[0], e=b[1]) for b in self.batches])
        else:
            return torch.stack(
                [si * self.eval(ki=ki, x=None) for ki, si in zip(range(self.k), self.s)]
            ).sum(dim=0)

    def batch_diff(self, prediction: torch.Tensor, target: torch.Tensor):
        diff = torch.zeros(1, dtype=torch.float32, requires_grad=True).to(
            torch.device("cuda:0")
        )
        for b in self.batches:
            diff += (prediction[b[0] : b[1]] - target[b[0] : b[1]]).pow(2).mean()

        return diff.mean()

    def __call__(self, batching: bool):
        return self.forward(batching)

    def generate_sdf_plot_data(self,spacing:float,min:float,max:float):

        def model_output(data:np.array,k:np.array,s:np.array):
            prediction = []
            for ki, si in zip(range(k),s):                        
                z = (self.eval(
                    ki=ki,
                    x=torch.from_numpy(data)
                    .type(torch.FloatTensor)
                    .to(torch.device("cuda:0")),
                ).cpu().detach().numpy())
                prediction.append(z* si)
                
            return np.asarray(prediction).sum()
                          
        points = int((abs(min)+abs(max)) / spacing) + 1
        
        s = self.s.cpu().detach().numpy()

        domain = np.linspace(min,max,points)
        x,y = np.meshgrid(domain,domain)
        z = []
        for xi,yi in zip(x.flatten(),y.flatten()):
            # d = np.asanyarray([[xi,yi] for _ in range(self.m)])
            d = np.asanyarray([[xi,yi]])   
            # output = model_output(data=d,k=self.k,s=s)
            # print(output)
            # zi = self.target.as_lambda('numpy')(xi,yi)
            zi = model_output(data=d,k=self.k,s=s)
            #print(f"x: {xi} y: {yi} z: {zi:.3f} sdf: {self.target.as_lambda('numpy')(xi,yi):.3f} (diff: {(abs(zi)-abs(self.target.as_lambda('numpy')(xi,yi))):.3f})")
            z.append(zi)

        return x,y,np.asanyarray(z).reshape(len(x),len(y))

    def print_params(self):
        print(
            f"Devices: \na: {self.a.device}\nb: {self.b.device} \nx: {self.x.device} \ns: {self.s.device}"
        )
