# Diffconvpl

Version: 0.1

## Installation of requirements
In order to be able to run the source code the required packages need to be installed.
This can be done via pip with the following command:

`pip install -r requirements_dev.txt`

In order to run the tests please install the required packages via

`pip install -r requirements_test.txt`

## Running the source code

Locate the `application.py` and simply run it with Python. This will run with the following default arguments

        APP_ARGS = {
            "func": "torch.sin(10*x)",
            "params": ["x"],
            "m": 32,
            "entries": 200000,
            "epochs": 5000,
            "positive_funcs": 1,
            "negative_funcs": 1,
            "fullplot": fullplot,
            
        }
