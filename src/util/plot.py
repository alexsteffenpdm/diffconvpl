import os

if os.getenv("TESTING") == "True":
    from common import rand_color
else:
    from .common import rand_color

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# SETUP_PY_ENV
BLENDER_PATH = os.path.normpath("C:\\Program Files\\Blender Foundation\\Blender 3.4")


def unscientific_plot(name: str, values: np.array):
    print(f"{name}: {np.array2string(values,formatter={'float':lambda x: f'{x:f}'})}\n")


def plotsdf(
    func: str,
    xv: np.array,
    yv: np.array,
    z: np.array,
    err_d: np.array,
    err_v: np.array,
    autosave: bool,
    losses: np.array,
    filename: str,
    display_blend: bool,
) -> None:
    assert len(z) ** 2 == (len(xv) * len(yv)), f"{len(z)} == ({len(xv)} * {len(yv)})"
    assert len(z.shape) == 2
    # fig = plt.figure("Results SDF", [10, 30])
    # err_x = np.arange(0, len(losses), 1)
    # plt.subplot(2, 2, 1)
    # plt.title("Error")
    # plt.plot(err_x, losses, color=rand_color(), label="Loss")
    # plt.xlabel("Iterations")
    # plt.ylabel("Error")
    # plt.legend(loc="best")

    # plt.subplot(2, 1, 2)

    # plt.contourf(xv, yv, z)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title(f"Approximation of {func}")
    # plt.legend(loc="best")
    # plt.axis("equal")

    # plt.subplot(2, 2, 2)
    # plt.title("Error Propagation")
    # plt.plot(err_d, err_v, color=rand_color(), label="Normalized Error")
    # plt.xlabel("x/y value")
    # plt.ylabel("SDF Error")
    # plt.legend(loc="best")

    # plt.savefig(f"data\\plots\\{filename}.png")
    # if autosave != False:
    #     plt.show(block=False)
    # plt.close()

    # fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax2.plot_surface(xv, yv, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig2.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    x = xv.flatten()
    y = yv.flatten()
    z = z.flatten()
    vertices = [[xi, yi, zi] for xi, yi, zi in zip(x, y, z)]

    with open("tmp.json", "w") as outfile:
        json.dump(
            {
                "mode": "create",
                "data": np.asanyarray(vertices).tolist(),
                "name": filename,
                "gridsize": [xv.shape[0], xv.shape[1]],
            },
            outfile,
        )
    print("STAGE: Data Export")
    # workaround for bpy import error, when handling subprocesses
    writer_path = os.path.join(os.getcwd(), "src", "blender", "scene_handler.py")
    print(os.popen(f"python {writer_path}").read())
    BLEND_PATH = os.path.join(
        os.getcwd(), "data", "generated", "blender_files", "scenes", f"{filename}.blend"
    )
    if display_blend == True:
        os.chdir(BLENDER_PATH)
        os.system(f"blender {BLEND_PATH}")
    return


def plot2d(
    func: str,
    x: np.ndarray,
    y_target: np.array,
    maxaffines: np.ndarray,
    y_pred: np.array,
    fullplot: bool,
    filename: str,
    autosave: bool,
    losses: np.array,
):
    unscientific_plot("x", x)
    unscientific_plot("y_target", y_target)
    unscientific_plot("maxaffines", maxaffines)
    unscientific_plot("y_pred", y_pred)

    fig = plt.figure("Results", [10, 20])
    plt.subplot(2, 1, 1)

    # original data
    x0 = x[:, 0]
    x1 = x[:, 1]
    plt.plot(x0, x1, color=rand_color(), label=func, linestyle="-.")

    # generated data
    if fullplot:
        for i, y_predi in enumerate(maxaffines):
            plt.plot(x0, y_predi, color=rand_color(), label=f"MaxAffine {i}")

    plt.plot(x0, y_pred, color=rand_color(), label="Prediction")

    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Approximation")
    plt.legend(loc="best")
    plt.axis("equal")

    err_x = np.arange(0, len(losses), 1)
    plt.subplot(2, 1, 2)
    plt.title("Error")
    plt.plot(err_x, losses, color=rand_color(), label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Error")

    plt.savefig(f"data\\plots\\{filename}.png")
    if autosave != False:
        plt.show(block=False)
        plt.show()
    plt.close()
    return
