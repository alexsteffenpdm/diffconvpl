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


The source code will now try to build a model that approximates the given `func`. Parameters can be changed to approximate other functions as well.

### Commandline Parameters
The program accepts certain commandline parameters:

`--autosave` will automatically save the generated output data without user-intercation.

`--fullplot` will add all evaluated MaxAffineFunctions to the output plot.

`--no-batch` will disable batched computation.

`--autorun <str:FILEPATH>` will recreate the model with parameters provided by a given filepath. Use the files within `data\json`.

`--batchsize <int:AMOUNT>` will set the size of batches for computation (only use, when large datasets are given)
