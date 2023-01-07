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

AUTOSAVE = False
APP_ARGS = {}


def plot_wrapper(
    func: str,
    x: np.array,
    y_target: np.array,
    maxaffines: np.ndarray,
    y_pred: np.array,
    fullplot: bool,
    timetag: str,
    autosave: bool,
    losses: np.array,
):
    plot2d(func, x, y_target, maxaffines, y_pred, fullplot, timetag, autosave, losses)


def run():
    global APP_ARGS, AUTOSAVE
    print("STAGE: Setup")
    fullplot = APP_ARGS["fullplot"]
    logger = ParamLogger()

    # setup params for MaxAffineFunction
    TARGET = Target(func=APP_ARGS["func"], parameters=APP_ARGS["params"])

    m = APP_ARGS["m"]
    entries = APP_ARGS["entries"]
    epochs = APP_ARGS["epochs"]
    positive_funcs = APP_ARGS["positive_funcs"]
    negative_funcs = APP_ARGS["negative_funcs"]

    batching = "batch_size" in APP_ARGS.keys()
    if batching:
        batch_size = APP_ARGS["batch_size"]

    # setup data
    signs = np.asarray(make_signs(positive=positive_funcs, negative=negative_funcs))
    k = len(signs)
    # datapoints = np.linspace(-1.0, 1.0, entries)
    # y = np.asarray([TARGET.as_lambda("torch")(torch.tensor(dp)) for dp in datapoints])
    datapoints = np.asanyarray([(d[0], d[1]) for d in APP_ARGS["data"]])
    y = np.asarray(
        [
            TARGET.as_lambda("torch")(torch.tensor(dp[0]), torch.tensor(dp[1]))
            for dp in datapoints
        ]
    )
    tensor_y = torch.from_numpy(y).type(torch.FloatTensor).to(torch.device("cuda:0"))
    area = torch.zeros_like(tensor_y).to(torch.device("cuda:0"))

    model = MultiDimMaxAffineFunction(
        target=TARGET,
        m=m,
        k=k,
        dim=2,
        x=datapoints,
        signs=signs,
        batchsize=batch_size if batching else 2**32,
    ).to(torch.device("cuda:0"))

    if batching == True:
        print("STAGE: Calculating batchsize")
        print_colored(
            "Warning: Running computation in batched mode is significantly slower. Please check whether it is needed!",
            cmdcolors.WARNING,
        )
        if APP_ARGS["batch_size"] == 2**32:
            model.bench(y_target=tensor_y, granularity=0.01)
            print(f"       Maximum applicaple batchsize={model.batch_size}")
        else:
            print(f"       Using preset batchsize of {batch_size}")
            model.batches = get_batch_spacing(batch_size, model.x.shape[0])

    print("STAGE: Calculation")

    optimizer = torch.optim.Adam([model.a, model.b], lr=0.06)

    loss = None
    loss_plot = []
    pbar = tqdm(total=epochs)

    for _ in range(epochs):
        optimizer.zero_grad()
        if batching == True:
            loss = model.batch_diff(model(batching), area)
        else:
            loss = (model(batching) - area).pow(2).mean()
        loss_plot.append(loss.item())
        loss.backward()

        optimizer.step()
        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()
    pbar_dict = pbar.format_dict

    print("STAGE: Plot")

    timetag = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    maxaffines, prediction = model.get_plot_data(batching, datapoints, y)
    plotprocess = multiprocessing.Process(
        target=plot_wrapper,
        args=(
            TARGET.no_package_str(),
            datapoints,
            y,
            maxaffines,
            prediction,
            fullplot,
            timetag,
            AUTOSAVE,
            loss_plot,
        ),
    )
    plotprocess.start()

    print("STAGE: Data")

    if AUTOSAVE == True:
        keep_data = True
        success = True
    else:
        keep_data = None
        success = None

    while keep_data is None:
        answer = input("Do you want to keep the used parameters? (y/n)\n")
        if answer in ["y", "n"]:
            keep_data = "y" == answer
        while success is None:
            answer = input("Is the fitting successful? (y/n)\n")
            if answer in ["y", "n"]:
                success = "y" == answer
                break

    if keep_data:
        log_dict = build_log_dict(
            tqdm_dict=pbar_dict,
            loss=loss.item(),
            func=TARGET.no_package_str(),
            positive=positive_funcs,
            negative=negative_funcs,
            success=success,
            autosave=AUTOSAVE,
        )
        logger.full_log(dict=log_dict)
        APP_ARGS["Success"] = success
        APP_ARGS["Autosave"] = AUTOSAVE
        logger.json_log(dict=APP_ARGS, filename=timetag)
    print("STAGE: End")


def setup(
    autosave: bool,
    fullplot: bool,
    filepath: str = None,
    batch_size: int = 2**32,
    no_batch: bool = False,
):
    global APP_ARGS, AUTOSAVE
    AUTOSAVE = autosave
    if not filepath:
        [
            os.makedirs(directory)
            for directory in ["data", "data\\json", "data\\plots","data\\generated"]
            if not os.path.exists(directory)
        ]

        # APP_ARGS START
        APP_ARGS = {
            "func": "torch.sin(10*x)",
            "params": ["x"],
            "m": 32,
            "entries": 100000,
            "epochs": 5000,
            "positive_funcs": 16,
            "negative_funcs": 16,
            "fullplot": fullplot,
        }
        # APP_ARGS STOP
        if no_batch == False:
            APP_ARGS["batch_size"] = batch_size

    else:
        APP_ARGS = rerun_experiment(filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Diffconvpl")
    parser.add_argument(
        "--autosave",
        action="store_true",
        help="Skips questions and directly saves data.",
    )
    parser.add_argument(
        "--fullplot",
        action="store_true",
        help="Enable plotting all max affine functions.",
    )
    parser.add_argument(
        "--no-batch", action="store_true", help="Disables batch-computation."
    )

    parser.add_argument(
        "--autorun",
        metavar=str,
        default=None,
        action="store",
        nargs="?",
        help="Issues and autorun with a given configuration via file.",
    )
    parser.add_argument(
        "--batchsize",
        metavar=int,
        default=2**32,
        action="store",
        nargs="?",
        help="Presets batch_size value used for large datasets",
    )

    args = parser.parse_args()

    setup(
        autosave=args.autosave,
        fullplot=args.fullplot,
        filepath=args.autorun,
        batch_size=int(args.batchsize),
        no_batch=args.no_batch,
    )
    run()
