import os
import json
import torch
import numpy as np
from typing import Tuple


# The class 'Initializer' initializes the a and b values for max-affine functions by
# parsing a JSON file with the following format:
#
# {
#     "name": <str>,        //  Arbitrary name for the initialization function
#     "symbol": <str>,      //  single character used for building an appropriate lambda function
#     "precision": <int>,   //  defines the decimal precision for the returned distance-values (only dim > 1)
#     "function": <str>,    //  function string (containing the defined 'symbol')
#     "derivative": <str>   //  function string for the derivative of 'function'
# }
#
# Remark: The initialized values are pseudo-random (generated via np.random.uniform)
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html (Version 1.24) (Last visited on: 23/02/13)


class Initializer(object):
    def __init__(self,filepath:str,dim:int=1) -> object:
        if not os.path.exists(filepath):
            raise ValueError(f"File does not exist under {filepath}")

        self.dim:int = dim
        name, symbol, precision, func, deriv = self.from_json(filepath)
        self.name:str = name
        self.symbol:str = symbol
        self.precision:int = int(eval(f"2* 1e{precision}"))

        self.func_str:str = f"lambda {self.symbol}: {func}"
        self.deriv_str:str = f"lambda {self.symbol}: {deriv}"

        self.func:str = eval(self.func_str)
        self.derivative:str = eval(self.deriv_str)
        self.domain:np.array = np.array([-1.0,1.0])


    def __repr__(self) -> str:
        return f"Parameter Initializer '{self.name}':\n\tSymbol: {self.symbol}\n\tFunction: {self.func_str}\n\tDerivative: {self.deriv_str}"


    def __call__(self,entries:int) -> Tuple[torch.Tensor,torch.Tensor]:
        samples:np.array = np.array([ np.random.uniform(self.domain[0],self.domain[1]) for _ in range(entries)])
        a:np.array = np.zeros_like(samples)
        b:np.array = np.zeros_like(samples)
        
        # Interpretation as arithmetic 1-dimensional function
        if self.dim == 1:

            a = np.asanyarray([ self.derivative(s) for s in samples])
            b = np.array([ (self.func(s) - self.derivative(s)*s) for s in samples])
            
        # Interpretation as distance function of an 1-dimensional arithmetic function
        elif self.dim == 2:

            a = np.asanyarray([ [self.derivative(s), (self.func(s) - self.derivative(s)*s)] for s in samples])

            # Here b defines the distance-value of the given samples
            b = np.array([ self.get_distance(ai) for ai in a])
           
        # Interpretation as distance function of a plane defined by a 1-dimensional arithmetic function and
        # a vector orthogonal to that function in spatial space
        elif self.dim == 3:

            # Here the y-value will define the plane, i.e. y has no influence on the distance-value, as the function
            # has no boundary along the y-axis (within the defined domain range from -1 to 1).
            a = np.asanyarray([ [self.derivative(s), s ,(self.func(s) - self.derivative(s)*s)] for s in samples])
            b = np.array([ self.get_distance(ai) for ai in a])

        else:
            raise ValueError(f"Evaluations for dimensions > 3 are not defined. (given: {self.dim})")
        return torch.from_numpy(a).type(torch.FloatTensor), torch.from_numpy(b).type(torch.FloatTensor)


    def from_json(self, filepath:str) -> Tuple[str,str,str,str,str]:
        with open(filepath,'r') as file:
            data = json.load(file)
            return data["name"], data["symbol"], data["precision"], data["function"], data["derivative"]

    def np_array_distance(self,point:np.array) -> float:
        distance = eval(f"lambda x: np.sqrt((x - ({point[0]}))**2 + (({self.func_str.split(': ')[1]}) - ({point[-1]}))**2)")
        return np.min(np.asarray([ distance(xi) for xi in np.linspace(self.domain[0],self.domain[1],self.precision)]))

    def torch_distance(self,point:np.array) -> float:
        t = torch.from_numpy(np.linspace(self.domain[0],self.domain[1],self.precision)).type(torch.FloatTensor).to(torch.device("cuda:0"))
        func = lambda t: torch.sqrt( torch.pow(t- point[0],2) + torch.pow(t**2 - point[1],2) )
        t = func(t)
        return torch.min(t,dim=-1).cpu().detach().numpy()

    def get_distance(self,point:np.array) -> float:

        # Calculation of the euclidean distance sqrt( (p_x - q_x)**2 - (p_y - q_y)** 2) where:
        # "p" is a point of the function defined in the JSON file
        # "q" is the point to which the distance-value is wanted  

        distance = eval(f"lambda x: np.sqrt((x - ({point[0]}))**2 + (({self.func_str.split(': ')[1]}) - ({point[-1]}))**2)")
        return np.min(np.asarray([ distance(xi) for xi in np.linspace(self.domain[0],self.domain[1],self.precision)]))

        # from time import time

        # t0 = time()
        # torch_solution = np.around(self.torch_distance(point),decimals=self.precision)
        # t1 = time()
        
        # t2 = time()
        # np_array_solution = np.around(self.np_array_distance(point),decimals=self.precision)
        # t3 = time()

        
        # print(f"torch took: {t1-t0} = {torch_solution}")
        # print(f"numpy took: {t3-t2} = {np_array_solution}")


if __name__ == "__main__":
    path = "src\\initializations\\parabola.json"
    p = Initializer(path,dim=2)

    p.get_distance(point=np.array([-0.3184,10.0]))