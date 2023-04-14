import json
import os
import random
from enum import Enum
from typing import Any, Optional


class cmdcolors(Enum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_colored(text: str, color: cmdcolors) -> None:
    print(f"{color} {text} {cmdcolors.ENDC}")


def rand_color() -> str:
    r = lambda: random.randint(0, 255)
    return f"#{r():02x}{r():02x}{r():02x}"


def make_signs(positive: int, negative: int) -> list[float]:
    assert positive >= 0
    assert negative >= 0
    assert (abs(positive) + abs(negative)) > 0
    signs = ([1.0] * positive) + ([-1.0] * negative)

    return signs


def build_log_dict(
    tqdm_dict: dict[str, Any],
    loss: float,
    func: str,
    positive: int,
    negative: int,
    success: Optional[bool],
    autosave: bool,
) -> dict[str, Any]:
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


def rerun_experiment(filepath: str = "") -> dict[str, Any]:
    if filepath:
        with open(filepath) as fp:
            return json.load(fp)
    try:
        selection = -1
        experiments = os.listdir("data\\json")
        while selection < 0:
            [print(f"{i}: {exp}") for i, exp in enumerate(experiments)]
            tmp = int(input(f"Select an experiment (0-{len(experiments)-1}): "))
            if tmp >= 0 and tmp <= len(experiments) - 1:
                selection = tmp

        with open(f"data\\json\\{experiments[selection]}") as fp:
            return json.load(fp)

    except:
        print("JSON directory not found")
        exit()


def get_batch_spacing(size: int, stop: int) -> list[tuple[int, int]]:
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
