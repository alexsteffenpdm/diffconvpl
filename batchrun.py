import os
import json
from tqdm import tqdm
from typing import Dict, Any
import argparse
import numpy as np
from datetime import datetime, timedelta
import itertools
import time


def parse_attrs(ns: argparse.Namespace, attrs: list[str]) -> argparse.Namespace:
    for attr in attrs:
        if hasattr(ns, attr):
            setattr(ns, attr, IterParam([int(v) for v in getattr(ns, attr)]))
    return ns


class IterParam:
    def __init__(self, values: list):
        self.start: int = values[0]
        self.stop: int = values[1] + values[2]
        self.step: int = values[2]

    def iter(self):
        return np.arange(self.start, self.stop, self.step)


class BatchRunner:
    def __init__(
        self,
        parameters: argparse.Namespace,
    ):
        self.itervalues: dict = {
            "m": parameters.M.iter().tolist(),
            "epochs": parameters.E.iter().tolist(),
            "positive_funcs": parameters.P.iter().tolist(),
            "negative_funcs": parameters.N.iter().tolist(),
            "iter_on": parameters.iter_on,
        }
        self.estimation, self.combinations = self.approximate_runtime(
            runtime_per_instance=240
        )

        self.filepath: str = os.path.join(os.getcwd(), parameters.fp)
        self.approximation_name: str = parameters.fp.split("\\")[-1].split(".")[0]
        self.full: bool = parameters.full
        self.data: dict[str, Any] = self.get_data()
        self.listen_to_dirs: dict[str, Any] = {
            "Autosave": os.path.join(os.getcwd(), "data", "json"),
            "Scene": os.path.join(
                os.getcwd(), "data", "generated", "blender_files", "scenes"
            ),
        }

    def approximate_runtime(self, runtime_per_instance: int) -> str:
        combinations = self.make_combinations()

        print(
            f"With this configuration there will be {len(combinations)} combinations.\nEstimated time per combination {str(timedelta(seconds=runtime_per_instance))} equates to: {str(timedelta(seconds=runtime_per_instance*len(combinations)))}"
        )
        ans = input("Do you wish to proceed? (y/n): ")
        if ans != "y":
            exit()
        return (
            str(timedelta(seconds=runtime_per_instance * len(combinations))),
            combinations,
        )

    def make_combinations(
        self,
    ) -> list:
        vars = []
        for k, v in self.itervalues.items():
            if k != "iter_on":
                if k not in self.itervalues["iter_on"]:
                    vars.append([v[-1]])
                else:
                    vars.append(v)
        return list(itertools.product(*vars))

    def get_data(
        self,
    ) -> dict[str, Any]:
        with open(self.filepath, "r") as fp:
            return json.load(fp)

    def get_filenames(
        self,
    ) -> dict[str, str]:
        filenames_dir = {}
        tag = ""
        with open(os.path.join(os.getcwd(), "current_timetag.txt"), "r") as tagfile:
            tag += tagfile.readline()
        for k, v in self.listen_to_dirs.items():
            for file in os.listdir(v):
                if tag in file:
                    filenames_dir[k] = file
        return filenames_dir

    def run(self):
        timetag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_log = f"BASE_MODEL: {self.filepath}\n"
        data_log += f"Total time: <TOTAL_TIME>\n\n"
        # each combination has format (m,epochs,positive_funcs,negative_funcs)
        total_start = datetime.now()
        for combination in self.combinations:
            self.data["m"] = combination[0]
            self.data["epochs"] = combination[1]
            self.data["positive_funcs"] = combination[2]
            self.data["negative_funcs"] = combination[3]
            print(f"\n\nUsing Combination: {combination} \n\n")
            with open(
                os.path.join(os.getcwd(), "runner_tmp.json"), "w"
            ) as runner_config:
                json.dump(self.data, runner_config, indent=4)
            start_approximation = datetime.now()
            data_log += f"Combination: m={combination[0]} epochs={combination[1]} positive_funcs={combination[2]} negative_funcs={combination[3]}\n"

            os.system(
                f"python .\\application.py --autorun {os.path.join(os.getcwd(),'runner_tmp.json')} --no-batch --autosave --blender-render False"
            )

            end_approximation = datetime.now()
            duration = str(
                timedelta(
                    seconds=(end_approximation - start_approximation).total_seconds()
                )
            )
            data_log += (
                f"Startdate: {start_approximation.strftime('%Y-%m-%d_%H-%M-%S')}\n"
            )
            data_log += f"Enddate: {end_approximation.strftime('%Y-%m-%d_%H-%M-%S')}\n"
            data_log += f"Duration: {duration}\n"
            data_log += f"Files:\n"
            print(f"\n\nUsing Combination: {combination} took {duration} \n\n")
            time.sleep(2)

            for k, v in self.get_filenames().items():
                data_log += f"\t{k}: {v}\n"
            data_log += "\n"
        total_end = datetime.now()
        total_duration = str(
            timedelta(seconds=(total_end - total_start).total_seconds())
        )
        data_log = data_log.replace("<TOTAL_TIME>", total_duration)
        data_log = data_log.rstrip()
        with open(
            os.path.join(
                os.getcwd(),
                "data",
                "runner_logs",
                f"{self.approximation_name}_{timetag}_log.txt",
            ),
            "w",
        ) as logfile:
            logfile.write(data_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batchrunner")

    parser.add_argument(
        "-fp",
        metavar="str",
        default=None,
        action="store",
        nargs="?",
        help="Specifies the filepath of the shape which SDF should be approximated with its parameters.",
    )

    parser.add_argument(
        "-full", action="store_true", help="Enables plotting all max-affine functions"
    )

    parser.add_argument(
        "-M",
        metavar=str,
        default=None,
        action="store",
        nargs="+",
        help="Specifies the used 'm' parameter of the application in the format of [min,max,step].",
    )

    parser.add_argument(
        "-P",
        metavar=str,
        default=None,
        action="store",
        nargs="+",
        help="Specifies the amount of positively weighted max-affine functions of the application in the format of [min,max,step].",
    )

    parser.add_argument(
        "-N",
        metavar=str,
        default=None,
        action="store",
        nargs="+",
        help="Specifies the amount of negativly weighted max-affine functions of the application in the format of [min,max,step].",
    )

    parser.add_argument(
        "-E",
        metavar=str,
        default=None,
        action="store",
        nargs="+",
        help="Specifies the amount of epochs of the application in the format of [min,max,step].",
    )

    parser.add_argument(
        "--iter-on",
        metavar=str,
        default=None,
        action="store",
        nargs="+",
        help="Defines the varabiles that should be changed by this program.",
    )

    # only for test
    # os.environ["DIFFCONV_PL_TIMETAG"]="2023-04-09-_08-05-04"

    args = parse_attrs(parser.parse_args(), ["E", "M", "N", "P"])
    runner = BatchRunner(parameters=args)
    runner.run()
