from .common import rand_color
import matplotlib.pyplot as plt
import numpy as np

def unscientific_plot(name: str, values: np.array):
    print(f"{name}: {np.array2string(values,formatter={'float':lambda x: f'{x:f}'})}\n")
         
def plotsdf(
    func: str,
    xv: np.ndarray,
    yv: np.ndarray,
    z: np.ndarray,
    autosave: bool,
    losses: np.array,
    filename: str,
):
    assert len(z)**2 == (len(xv) * len(yv)),f"{len(z)} == ({len(xv)} * {len(yv)})"
    assert len(z.shape) == 2
    
    fig = plt.figure("Results SDF", [10, 20])
    plt.subplot(2, 1, 1)   
   
    plt.contourf(xv,yv,z,label=func)
    plt.xlabel("x")
    plt.ylabel("y")
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
    unscientific_plot("x",x)
    unscientific_plot("y_target",y_target)
    unscientific_plot("maxaffines",maxaffines)
    unscientific_plot("y_pred",y_pred)



    fig = plt.figure("Results", [10, 20])
    plt.subplot(2, 1, 1)

    # original data
    x0 = x[:,0]
    x1 = x[:,1]
    plt.plot(x0,x1, color=rand_color(), label=func, linestyle="-.")

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
