from .util.common import rand_color
from .util.target import Target

import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import List


class MultiDimMaxAffineFunction(torch.nn.Module):
    def __init__(self,m:int,k:int,dim:int,x,signs:List[float],target:Target):
        super().__init__()    

        self.target = target

        self.x = torch.nn.Parameter(torch.from_numpy(x),requires_grad=False).type(torch.FloatTensor).to(torch.device("cuda:0"))

        self.s = torch.nn.Parameter(torch.from_numpy(signs),requires_grad=False) #.to(torch.device("cuda:0"))
    
        self.a = torch.nn.Parameter(torch.rand((k,m,dim),dtype=torch.float32),requires_grad=True) #.to(torch.device("cuda:0"))
        
        self.b = torch.nn.Parameter(torch.rand((k,m),dtype=torch.float32),requires_grad=True) #.to(torch.device("cuda:0"))
        self.k = k
        self.dim = dim
        self.m = m
        self.domains = np.linspace(0.0,1.0,self.k+1)

       # self.func= lambda t: torch.max(t,dim=-1)[0]
        self.func = lambda t: torch.logsumexp(t,dim=-1)

        self.param_init()
        #self.randomize()

    def randomize(self):
        domain = [-1,1]
        xi = torch.rand_like(self.a[0])
        xi = (domain[1]- domain[0])*xi + domain[0]

        a = 1
        b = 0
        for ki in range(self.k):
            self.a[ki].data = a*2*xi
            self.b[ki].data = a*(xi**2).sum(dim=1) + b - (self.a.data*xi).sum(dim=-1)

    def param_init(self):

        def _gauss_dist(s:int,e:int,entries:int):
            #dist = np.linspace(s,e,entries)
            dist = np.linspace(-1.0,1.0,entries)
            domain_range = (np.max(dist) + (-1*np.min(dist) if np.min(dist) < 0.0 else np.min(dist)))
            s = []
            for d in dist:
                v = (2*abs(d) / domain_range)
                if d < 0.0:
                    s.append(-v)
                else:
                    s.append(v)
            return s

        domains = np.linspace(-1.0,1.0,self.k+1)

        Arr_a = []
        Arr_b = []
        for ki in range(self.k):
            a_ki = []
            b_ki = []
            for xi in _gauss_dist(domains[ki],domains[ki+1],entries=self.m):
                xi = torch.from_numpy(np.asarray(xi)).requires_grad_(True)
                #xi = xi.clone().detach()
                y = self.target.as_lambda("torch")(xi)
                y.backward()            

                a_ki.append(xi.grad.item())
                b_ki.append(y.item() - (xi.grad.item() *xi.item()))
            Arr_a.append(a_ki)
            Arr_b.append(b_ki)
          
        with torch.no_grad():
            for ki in range(self.k):
                for a_ki,b_ki in zip(Arr_a,Arr_b):
                    for i in range(len(a_ki)):
                        self.a[ki,i] = a_ki[i]
                        self.b[ki,i] = b_ki[i]
  
            assert self.a.shape == (self.k,self.m,self.dim)
       


    def eval(self,ki,x=None):
        if len(self.x.shape) == 1:
            self.x = self.x[:, None]
        if x == None: 
            return self.func(self.x @ self.a[ki].T + self.b[ki])
        else:
            return self.func(x @ self.a[ki].T + self.b[ki])
       

    def forward(self):
        return torch.stack([si*self.eval(ki=ki,x=None) for ki,si in zip(range(self.k),self.s)]).sum(dim=0)

    def __call__(self):
        return self.forward()

    def get_plot_data(self,x,y_target):
        y_pred = []
        for ki,si in zip(range(self.k),self.s):
            y_xi = []    
            for xi in x:
                y_calc = self.eval(
                            ki=ki,
                            x=torch.from_numpy(np.asarray([xi])).type(torch.FloatTensor).to(torch.device("cuda:0"))
                        ).cpu().detach().numpy()
                s_calc = si.cpu().detach().numpy()

                y_xi.append(s_calc*(y_calc+ki))
            y_pred.append(y_xi)
      
        y_calc = self.forward().cpu().detach().numpy()     

        return y_pred, y_calc


    def print_params(self):
        print(f"Devices: \na: {self.a.device}\nb: {self.b.device} \nx: {self.x.device} \ns: {self.s.device}")