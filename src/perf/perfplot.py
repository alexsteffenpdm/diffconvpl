import matplotlib.pyplot as plt
import numpy as np
import csv

EXP_FILE = "data\\experiments.csv"


def perfomance_plot(func: str, m: int):
    x = []
    y_duration = []
    y_iters = []
    y_error = []
    with open(EXP_FILE, "r", newline="") as fp:
        for row in csv.DictReader(fp, delimiter=","):
            if row["Function"] == func and row["Autosave"]:
                x.append(m)
                m += 50
                y_duration.append(float(row["Duration"]))
                y_iters.append(float(row["Iters per Second"]))
                y_error.append(float(row["Error"]))

    plt.figure(10, [10, 20])

    plt.subplot(3, 1, 1)
    plt.plot(x, y_duration)
    plt.title("Duration")
    plt.xlabel("m")
    plt.ylabel("Time in S")

    plt.subplot(3, 1, 2)
    plt.plot(x, y_iters)
    plt.title("Iterations per second")
    plt.xlabel("m")
    plt.ylabel("Iterations")

    plt.subplot(3, 1, 3)
    plt.plot(x, y_error)
    plt.title("Error")
    plt.xlabel("m")
    plt.ylabel("Error")

    plt.show()


if __name__ == "__main__":
    perfomance_plot(func="sin(10*x)", m=50)
