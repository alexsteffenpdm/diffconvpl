import os
import sys
import json
from tqdm import tqdm
from typing import Dict, Any

JSON_PATH = "data\\json\\batchrun.json"
APP_ARGS = {
    "func": "torch.sin(10*x)",
    "params": ["x"],
    "m": 2,
    "entries": 5000,
    "epochs": 20000,
    "positive_funcs": 4,
    "negative_funcs": 4,
    "fullplot": False,
}


def batchrun_json(app_args: Dict[str, Any]):
    global JSON_PATH, APP_ARGS
    if os.path.exists(JSON_PATH):
        os.remove(JSON_PATH)

    with open(JSON_PATH, "w") as fp:
        json.dump(APP_ARGS, fp, ensure_ascii=True, indent=2)


def run_diffconvpl():
    os.system(
        f'"C:\\Users\\alexf\\AppData\\Local\\Programs\\Python\\Python310\\python.exe application.py" -rerun -autosave --autorun {JSON_PATH}'
    )


def run():
    global APP_ARGS
    entries = APP_ARGS["entries"]
    for i in range(50, entries, 50):
        print(f"\n\nRunning approximation of {APP_ARGS['func']} with m={i}\n\n")
        APP_ARGS["m"] = i
        batchrun_json(app_args=APP_ARGS)
        run_diffconvpl()


if __name__ == "__main__":
    run()
