from src.maxaffine import MultiDimMaxAffineFunction
from src.util.common import *
from src.util.logger import ParamLogger
from src.util.plot import plot2d
from src.util.target import Target

import numpy as np
import os
import torch
import random
import multiprocessing
import argparse

from tqdm import tqdm
from datetime import datetime

random.seed(1)


APP_ARGS = {}

def plot_wrapper(func:str,x:np.array,y_target:np.array,maxaffines:np.ndarray,y_pred:np.array,fullplot:bool,timetag:str):
    plot2d(func,x,y_target,maxaffines,y_pred,fullplot,timetag)

"""
    APP_ARGS = {
            "func": "torch.sin(10*x)",
            "params": ["x"],
            "m": 7,
            "entries": 1000,
            "epochs": 20000,
            "positive_funcs": 4,
            "negative_funcs": 4,
            "fullplot": False,
        }
"""


def run():

    print("STAGE: Setup")
    # setup plot data
    fullplot=APP_ARGS["fullplot"]

    # setup logger
    logger = ParamLogger()

    # setup params for MaxAffineFunction
    TARGET = Target(func=APP_ARGS["func"], parameters=APP_ARGS["params"])

    m = APP_ARGS["m"]
    entries = APP_ARGS["entries"]
    epochs = APP_ARGS["epochs"]
    positive_funcs = APP_ARGS["positive_funcs"]
    negative_funcs = APP_ARGS["negative_funcs"]


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

    optimizer = torch.optim.Adam([model.a,model.b],lr = 0.06)

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
    
    timetag = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    maxaffines,prediction = model.get_plot_data(datapoints,y)
    plotprocess = multiprocessing.Process(target=plot_wrapper,args=(TARGET.no_package_str(),datapoints,y,maxaffines,prediction,fullplot,timetag))
    plotprocess.start()

    print("STAGE: Data")
    keep_data = None
    success = None
    while keep_data is None or success is None:
        if keep_data is None:
            answer = input("Do you want to keep the used parameters? (y/n)\n")
            if answer in ["y","n"]:
                keep_data = "y" == answer
        if success is None:
            answer = input("Is the fitting successful? (y/n)\n")
            if answer in ["y","n"]:
                success = "y" == answer
    
    if keep_data:
        log_dict = build_log_dict(
            tqdm_dict=pbar_dict,
            loss = loss.item(),
            func = TARGET.no_package_str(),
            positive=positive_funcs,
            negative=negative_funcs,
            success=success,
        )
        logger.full_log(dict=log_dict)
        APP_ARGS["Success"] = success
        logger.json_log(dict=APP_ARGS,filename=timetag)

def setup(rerun:bool):
    global APP_ARGS
    if rerun == False:
        [os.makedirs(directory) for directory in ["data","data\\json","data\\plots"] if not os.path.exists(directory)] 

        
        APP_ARGS = {
            "func": "x**2",
            "params": ["x"],
            "m": 2,
            "entries": 1000,
            "epochs": 20000,
            "positive_funcs": 1,
            "negative_funcs": 0,
            "fullplot": False,
        }
    else:
        APP_ARGS = rerun_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffconvpl"
    )
    parser.add_argument('-rerun',action='store_true',help='Select an experiment to rerun from "data\json\\"')
    args = parser.parse_args()
  
    setup(rerun=args.rerun)
    run()

