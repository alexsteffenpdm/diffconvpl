from typing import Dict, Any
import random
import os
import json


class cmdcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_colored(text: str, color: cmdcolors):
    print(color + text + cmdcolors.ENDC)


def rand_color():
    r = lambda: random.randint(0, 255)
    return "#{:02x}{:02x}{:02x}".format(r(), r(), r())


def make_signs(positive: int, negative: int):
    assert positive >= 0
    assert negative >= 0
    assert (abs(positive) + abs(negative)) > 0

    return ([1.0] * positive) + ([-1.0] * negative)


def build_log_dict(
    tqdm_dict: Dict[str, Any],
    loss: float,
    func: str,
    positive: int,
    negative: int,
    success: bool,
    autosave: bool,
):

    return {
        "Function": func,
        "Iterations": str(tqdm_dict["total"]),
        "Positive": str(positive),
        "Negative": str(negative),
        "Error": str(loss),
        "Duration": str(tqdm_dict["elapsed"]),
        "Iters per Second": str((tqdm_dict["total"]) / tqdm_dict["elapsed"]),
        "Success": success,
        "Autosave": autosave,
    }


def rerun_experiment(filepath: str = None):
    if filepath:
        with open(filepath, "r") as fp:
            return json.load(fp)
    try:

        selection = -1
        experiments = os.listdir("data\\json")
        while selection < 0:
            [print(f"{i}: {exp}") for i, exp in enumerate(experiments)]
            tmp = int(input(f"Select an experiment (0-{len(experiments)-1}): "))
            if tmp >= 0 and tmp <= len(experiments) - 1:
                selection = tmp

        with open(f"data\\json\\{experiments[selection]}", "r") as fp:
            return json.load(fp)

    except:
        print("JSON directory not found")
        pass


def get_batch_spacing(size, stop):
    x = size - 1
    arr = [(0, size)]
    while x < stop:
        s = x
        x += size
        if x <= stop:
            arr.append((s, x))
            if x == stop - 1:
                break
        else:
            arr.append((s, stop - 1))
    return arr
