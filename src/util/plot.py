from .common import rand_color
from .target import Target
import matplotlib.pyplot as plt
import numpy as np

from typing import Any


def plot2d(func:str,x:np.array,y_target:np.array,maxaffines:np.ndarray,y_pred:np.array,fullplot:bool,filename:str):
    
    if fullplot:
        for i,y_predi in enumerate(maxaffines):
            plt.plot(x,y_predi,color=rand_color(),label=f"MaxAffine {i}")

    plt.plot(x,y_pred,color=rand_color(),label="Prediction")
    plt.plot(x,y_target,color=rand_color(),label=func,linestyle="-.")
    plt.legend(loc="best")
    plt.savefig(f"data\\plots\\{filename}.png")
    plt.show()
    
    return 

