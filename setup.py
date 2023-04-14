import os
import platform
import sys
import subprocess


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

STDOUT = True

try:
    if sys.argv[1] == "get_output":
        STDOUT = False
except:
    pass


def printl(value: str) -> None:
    print(f"{value} ", end="")


def wrap_format(format: str, value: str) -> str:
    return format + value + cmdcolors.ENDC


################# INTRO TEXT #################


################# PLATFORM CHECK #################
printl(f"Checking Platform:")
_platform_ = platform.system()
printl(_platform_)
if _platform_ != "Windows":
    print(
        f"{wrap_format(cmdcolors.BOLD + cmdcolors.WARNING,f'This software has only been tested on Windows.')}"
    )
else:
    print(f"{wrap_format(cmdcolors.OKGREEN,'OK')}")

################# PYVERSION CHECK #################
printl(f"Checking Python Version:")
_py_version_ = f"{sys.version_info.major}.{sys.version_info.minor}"
printl(_py_version_)
if _py_version_ != "3.10":
    printl(wrap_format(cmdcolors.FAIL, "NOK"))
    print(
        f"{wrap_format(cmdcolors.FAIL + cmdcolors.BOLD,f'Python Version 3.10.* is required to run and install all required packages. Aborting!')}"
    )
    exit(1)
else:
    print(f"{wrap_format(cmdcolors.OKGREEN,'OK')}")


################# ENVFILE CHECK #################
printl(f"Checking environment file:")
environment = {}
with open(os.path.join(os.getcwd(), ".env"), "r") as fp:
    for line in fp.readlines():
        line = line.replace('"', "")
        line_contents = line.split("=")
        environment[line_contents[0]] = line_contents[1].strip()

errors = []
for k, v in environment.items():
    match k:
        case "BLENDER_BINARY_PATH":
            if os.path.exists(rf"{v}") is not True:
                errors.append(
                    f"{k}: Given path for the blender executable does not exist."
                )
        case "DIFFCONVPL_TIMETAG":
            if not v:
                errors.append(f"{k}: No value given for the timetag.")
        case "GLOBAL_SEED":
            if int(v) <= 0:
                errors.append(f"{k}: Given global seed is not positive.")
        case "VIRTUAL_ENVIRONMENT_NAME":
            if len(v) <= 0:
                errors.append(
                    f"{k}: Given name for the virtual environment is invalid."
                )
        case _:
            errors.append(
                f"{k}: Unknown environment variable with value '{v}' is specified. Please remove."
            )

if len(errors) != 0:
    print(wrap_format(cmdcolors.FAIL, "NOK"))
    for e in errors:
        print(e)
    exit(1)
else:
    print(f"{wrap_format(cmdcolors.OKGREEN,'OK')}")

_venv_interpreter_path_ = ""
_venv_activate_script_path_ = ""


################# VIRTUAL ENV CREATION #################
printl("Creating virtual environment:")

try:
    _venv_interpreter_path_ = os.path.join(
        os.getcwd(), environment["VIRTUAL_ENVIRONMENT_NAME"], "Scripts", "python.exe"
    )
    _venv_activate_script_path_ = os.path.join(
        os.getcwd(), environment["VIRTUAL_ENVIRONMENT_NAME"], "Scripts", "activate"
    )

    subprocess.run(
        f"python.exe -m venv {environment['VIRTUAL_ENVIRONMENT_NAME']}",
        capture_output=STDOUT,
        shell=True,
    )
except:
    print("Error while creating virtual environment.")
    exit(1)
finally:
    print(f"{wrap_format(cmdcolors.OKGREEN,'OK')}")

################# PIP UPGRADE #################
printl("Upgrading pip:")
try:
    subprocess.run(
        f"{_venv_interpreter_path_} -m pip install --upgrade pip",
        capture_output=STDOUT,
        shell=True,
    )
except:
    print("Error while installing requirements.")
    exit(1)

finally:
    print(f"{wrap_format(cmdcolors.OKGREEN,'OK')}")


################# PACKAGE INSTALLATION #################
printl("Installing requirements via pip: ")
try:
    subprocess.run(
        f"{_venv_activate_script_path_+'.bat'} & pip install -r requirements_dev.txt",
        capture_output=STDOUT,
        shell=True,
    )
except:
    print("Error while installing requirements.")
    exit(1)

finally:
    print(f"{wrap_format(cmdcolors.OKGREEN,'OK')}")


################# SETTING ENV VARIABLES #################
_venv_activate_script_content_ = []
printl("Setting required environment variables:")
try:
    with open(_venv_activate_script_path_ + ".ps1", "r") as fp:
        for line in fp.readlines():
            _venv_activate_script_content_.append(line)

    with open(_venv_activate_script_path_ + ".ps1", "w") as fp:
        for line in _venv_activate_script_content_:
            if line == "# SIG # Begin signature block\n":
                for k, v in environment.items():
                    fp.write(f"$Env:{k} = '{v}'\n")
                fp.write(f"\n# SIG # Begin signature block\n")
            fp.write(line)

except:
    print("Error while setting environment variables.")
    exit(1)

finally:
    print(f"{wrap_format(cmdcolors.OKGREEN,'OK')}")


################# OUTRO TEXT #################
