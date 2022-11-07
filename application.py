from src.maxaffine import MultiDimMaxAffineFunction
from src.util.common import *
from src.util.logger import ParamLogger
from src.util.plot import plot2d
from src.util.target import Target

import numpy as np
import torch
import random
import multiprocessing

from tqdm import tqdm

random.seed(1)

def plot_wrapper(func:str,x:np.array,y_target:np.array,maxaffines:np.ndarray,y_pred:np.array,fullplot:bool):
    plot2d(func,x,y_target,maxaffines,y_pred,fullplot)



def run():
    print("STAGE: Setup")
    # setup plot data
    fullplot=True

    # setup logger
    logger = ParamLogger()

    # setup params for MaxAffineFunction
    TARGET = Target(func="torch.sin(5*x)", parameters=["x"])

    m = 500
    entries = 1000
    epochs = 20000
    positive_funcs = 2
    negative_funcs = 2


    # setup data
    
    signs = np.asarray(make_signs(positive=positive_funcs,negative=negative_funcs))
    k = len(signs)
    datapoints = np.linspace(-1.0,1.0,entries)
    y = np.asarray([TARGET.as_lambda("torch")(torch.tensor(dp)) for dp in datapoints])
    tensor_y = torch.from_numpy(y).type(torch.FloatTensor).to(torch.device("cuda:0"))


    model = MultiDimMaxAffineFunction(
        target=TARGET,
        m=m,
        k=k,
        dim=1,
        x=datapoints,
        signs=signs
    ).to(torch.device("cuda:0"))

    print("STAGE: Calculation")

    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01, momentum=0.9) #.to(torch.device("cuda:0"))

    loss = None
    pbar = tqdm(total=epochs)

    for _ in range(epochs):
        optimizer.zero_grad()

        loss = (model() - tensor_y).pow(2).mean()
        loss.backward()

        optimizer.step()
        pbar.update(1)
    pbar.close()
    pbar_dict = pbar.format_dict

    print("STAGE: Plot")

    maxaffines,prediction = model.get_plot_data(datapoints,y)
    plotprocess = multiprocessing.Process(target=plot_wrapper,args=(TARGET.no_package_str(),datapoints,y,maxaffines,prediction,fullplot))
    plotprocess.start()

    print("STAGE: Data")
    keep_data = None
    while True:
        answer = input("Do you want to keep the used parameters? (y/n)")
        if answer in ["y","n"]:
            keep_data = "y" == answer
            break

    if keep_data:
        log_dict = build_log_dict(
            tqdm_dict=pbar_dict,
            loss = loss.item(),
            func = TARGET.no_package_str(),
            positive=positive_funcs,
            negative=negative_funcs
        )
        logger.full_log(dict=log_dict)


if __name__ == "__main__":
    run()

