import argparse
import os
import random
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.models.maxaffine import MultiDimMaxAffineFunction
from src.util.common import *
from src.util.logger import ParamLogger
from src.util.parameter_initializer import Initializer
from src.util.plot import plotsdf
from src.util.shm_process import MemorySharedSubprocess

random.seed(os.environ.get("GLOBAL_SEED"))

AUTOSAVE: bool = False
APP_ARGS: dict[str, Any] = {}


def run():
    global APP_ARGS, AUTOSAVE
    print("STAGE: Setup")
    fullplot = APP_ARGS["fullplot"]
    logger = ParamLogger()

    m = APP_ARGS["m"]
    entries = APP_ARGS["entries"]
    epochs = APP_ARGS["epochs"]
    positive_funcs = APP_ARGS["positive_funcs"]
    negative_funcs = APP_ARGS["negative_funcs"]
    display = APP_ARGS["display_blender"]

    batching = "batch_size" in APP_ARGS.keys()
    if batching:
        batch_size = APP_ARGS["batch_size"]

    signs: np.array = np.asarray(
        make_signs(positive=positive_funcs, negative=negative_funcs)
    )
    k: int = len(signs)

    datapoints: np.array = np.asanyarray([(d[0], d[1]) for d in APP_ARGS["data"]])

    y: torch.Tensor = (
        torch.from_numpy(np.asarray([dp[2] for dp in APP_ARGS["data"]]))
        .type(torch.FloatTensor)
        .to(torch.device("cuda:0"))
    )

    model: MultiDimMaxAffineFunction = MultiDimMaxAffineFunction(
        m=m,
        k=k,
        dim=2,
        x=datapoints,
        signs=signs,
        initializer=Initializer("src\\initializations\\parabola.json", dim=2),
        batchsize=batch_size if batching else 2**32,
    ).to(torch.device("cuda:0"))

    if batching == True:
        print("STAGE: Calculating batchsize")
        print_colored(
            "Warning: Running computation in batched mode is significantly slower. Please check whether it is needed!",
            cmdcolors.WARNING,
        )
        if APP_ARGS["batch_size"] == 2**32:
            model.bench(y_target=y, granularity=0.01)
            print(f"       Maximum applicaple batchsize={model.batch_size}")
        else:
            print(f"       Using preset batchsize of {batch_size}")
            model.batches = get_batch_spacing(batch_size, model.x.shape[0])

    print("STAGE: Calculation")

    optimizer: torch.optim = torch.optim.Adam([model.a, model.b], lr=0.06)
    loss: torch.Tensor = None
    loss_plot: list[float] = []
    pbar = tqdm(total=epochs)
    if batching == True:
        for _ in range(epochs):
            optimizer.zero_grad()

            loss = model.batch_diff(model(batching), y)
            loss_plot.append(loss.item())
            loss.backward()

            optimizer.step()
            torch.cuda.empty_cache()
            pbar.update(1)
    else:
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = (model(batching) - y).pow(2).mean()
            loss_plot.append(loss.item())
            loss.backward()

            optimizer.step()
            pbar.update(1)

    pbar.close()
    pbar_dict = pbar.format_dict

    print("STAGE: Model Evaluation")

    timetag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(os.getcwd(), "current_timetag.txt"), "w") as fp:
        fp.write(timetag)
        fp.close()

    spacing = 0.01
    points: np.array = int((abs(-1.0) + abs(1.0)) / spacing) + 1
    domain: np.array = np.linspace(-1.0, 1.0, points)
    x, y = np.meshgrid(domain, domain)

    z: np.ndarray = np.zeros_like(x)
    for ki in tqdm(range(model.k)):
        z += model.generate_sdf_plot_data_single_maxaffine_function_vectorized(
            x=x, y=y, k=ki
        )

    # try:
    #     error_domain, error_values = model.error_propagation(
    #         spacing=0.01, min=-20.0, max=20.0
    #     )
    # except:
    #     print("ERROR while generating error propagation")

    plot_dict = {
        "func": "Unkown",
        "xv": x.tolist(),
        "yv": y.tolist(),
        "z": z.tolist(),
        "err_d": [],
        "err_v": [],
        "autosave": AUTOSAVE,
        "losses": loss_plot,
        "filename": timetag,
        "display_blend": display,
    }

    with open(
        os.path.join(
            os.getcwd(),
            "data",
            "generated",
            "blender_files",
            f"datapoints_{timetag}.txt",
        ),
        "w",
    ) as dp_file:
        for dp in APP_ARGS["data"]:
            dp_file.write(f"{dp[0]:.4f} {dp[1]:.4f} {dp[2]:.4f}\n")

    shared_memory_process = MemorySharedSubprocess(target=plotsdf)
    shared_memory_process.fork_on(data=plot_dict)
    shared_memory_process.await_join()

    print("STAGE: End")

    keep_data: Optional[bool]
    success: Optional[bool]
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
            func="Unknown",
            positive=positive_funcs,
            negative=negative_funcs,
            success=success,
            autosave=AUTOSAVE,
        )
        logger.full_log(dict=log_dict)
        APP_ARGS["Success"] = success
        APP_ARGS["Autosave"] = AUTOSAVE
        logger.json_log(dict=APP_ARGS, filename=timetag)

    if os.environ.get("diffconvpl_running") is not None:
        os.environ["diffconvpl_running"] = "0"


def setup(
    autosave: bool,
    fullplot: bool,
    filepath: Optional[str] = None,
    batch_size: int = 2**32,
    no_batch: bool = False,
    display_blender: bool = False,
) -> None:
    global APP_ARGS, AUTOSAVE
    AUTOSAVE = autosave
    if not filepath:
        for directory in ["data", "data\\json", "data\\plots", "data\\generated"]:
            if not os.path.exists(directory):
                os.makedirs(directory)

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

        if no_batch == False:
            APP_ARGS["batch_size"] = batch_size

    else:
        APP_ARGS = rerun_experiment(filepath)
    APP_ARGS["display_blender"] = display_blender


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
        help="Enable plotting all max-affine functions.",
    )

    parser.add_argument(
        "--no-batch", action="store_true", help="Disables batch-computation."
    )

    parser.add_argument(
        "--autorun",
        metavar="str",
        default=None,
        action="store",
        nargs="?",
        help="Issues and autorun with a given configuration via file.",
    )

    parser.add_argument(
        "--batchsize",
        metavar="int",
        default=2**32,
        action="store",
        nargs="?",
        help="Presets batch_size value used for large datasets",
    )

    parser.add_argument(
        "--blender-render",
        metavar="bool",
        default=False,
        action="store",
        nargs="?",
        help="Wether to run blender and display the results after computation.",
    )
    args = parser.parse_args()

    setup(
        autosave=args.autosave,
        fullplot=args.fullplot,
        filepath=args.autorun,
        batch_size=int(args.batchsize),
        no_batch=args.no_batch,
        display_blender=args.blender_render,
    )
    run()
