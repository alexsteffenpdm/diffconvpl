import csv
import json
import logging
import os
from typing import Dict


class ParamLogger:
    def __init__(
        self,
    ):
        self.logger = logging.getLogger("DiffconvPL_Logger")
        self.logger.setLevel(logging.DEBUG)

        self.filehandler = logging.FileHandler(r"data\experiments.log")
        self.filehandler.setLevel(logging.DEBUG)

        self.logger.addHandler(self.filehandler)

        self.formatter = logging.Formatter("%(message)s")

        self.csv = r"data\experiments.csv"

        self.fields = [
            "Function",
            "Iterations",
            "Positive",
            "Negative",
            "Error",
            "Duration",
            "Iters per Second",
            "Success",
            "Autosave",
        ]

    def log_entry(
        self,
        dict: dict[str, str],
    ):
        msg = ""
        for key, value in dict.items():
            msg += f"{key}: {value}\n"
        self.logger.debug(msg=msg)

    def csv_log(self, dict: dict[str, str]):
        if os.path.exists(self.csv):
            with open(self.csv, "a", encoding="UTF8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.fields)
                writer.writerow(dict)
        else:
            with open(self.csv, "w", encoding="UTF8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.fields)
                writer.writeheader()
                writer.writerow(dict)

    def full_log(self, dict: dict[str, str]):
        self.log_entry(dict=dict)
        self.csv_log(dict=dict)

    def json_log(self, dict: dict[str, str], filename: str):
        if dict["Autosave"] == True:
            filepath = f"data\\json\\AUTOSAVE_{filename}.json"
        else:
            if dict["Success"] == True:
                filepath = f"data\\json\\SUCCESS_{filename}.json"
            else:
                filepath = f"data\\json\\FAILED_{filename}.json"

        del dict["Success"]
        dict["plot_data"] = f"data\\plots\\{filename}.png"

        with open(filepath, "w") as fp:
            json.dump(dict, fp, ensure_ascii=True, indent=2)
