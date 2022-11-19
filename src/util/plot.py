from .common import rand_color
import matplotlib.pyplot as plt
import numpy as np


def plot2d(func:str,x:np.array,y_target:np.array,maxaffines:np.ndarray,y_pred:np.array,fullplot:bool,filename:str,autosave:bool,losses:np.array):
    
    fig = plt.figure("Results",[10,20])
    plt.subplot(2,1,1)


    if fullplot:
        for i,y_predi in enumerate(maxaffines):
            plt.plot(x,y_predi,color=rand_color(),label=f"MaxAffine {i}")



    plt.plot(x,y_pred,color=rand_color(),label="Prediction")
    plt.plot(x,y_target,color=rand_color(),label=func,linestyle="-.")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Approximation")
    plt.legend(loc="best")
    
  

    err_x = np.arange(0,len(losses),1)
    plt.subplot(2,1,2)
    plt.title("Error")
    plt.plot(err_x,losses,color=rand_color(),label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Error")

    plt.savefig(f"data\\plots\\{filename}.png")
    if autosave !=False:
        plt.show(block=False)    
        plt.show()
    plt.close()   
    return 

