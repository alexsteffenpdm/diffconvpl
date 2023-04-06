# Diffconvpl
Version: <VERSION>
##
As of version 0.2.0 this project requires Python 3.10.* .
***
## General
***



### Installation of requirements
In order to be able to run the source code the required packages need to be installed.
This can be done via pip with the following command:

`pip install -r requirements_dev.txt`

In order to run the tests please install the required packages via

`pip install -r requirements_test.txt`

### Running the source code

Locate the `application.py` and simply run it with Python. This will run with the following default arguments

<DEFAULT_ARGS>

The source code will now try to build a model that approximates the given `func`. Parameters can be changed to approximate other functions as well.

### Commandline Arguments

The program accepts certain commandline Arguments:

`--autosave` will automatically save the generated output data without user-intercation.

`--fullplot` will add all evaluated MaxAffineFunctions to the output plot.

`--no-batch` will disable batched computation.

`--autorun <str:FILEPATH>` will recreate the model with parameters provided by a given filepath. Use the files within `data\json`.

`--batchsize <int:AMOUNT>` will set the size of batches for computation (only use, when large datasets are given)

## SDF Generators
***

The branch `sdf2d` contains scripts that generate `JSON`-Files that can be applied to the `application.py` using the `--autorun` command line argument.
Located under **`<GENERATOR_LOCATION>`** the scripts will produce the aforementioned `JSON`-Files under **`<GENERATED_JSON_LOCATION>`** .

### Example call

`python .\application.py --autorun data\generated\2DSDF_Rectangle_BETA.json --no-batch --autosave`

### Generators and Commandline Arguments

The following generators exist and provide the functionality to parse specific commandline arguments as shown below:

*Note: The square generator can only handle datapoints in increments of 4.*
*Note: The triangle generator can only handle datapoints in increments of 3.*

<GENERATOR_HELP_TEXTS>

## Findings

With the current setup ( as of 2023/01/12 ) the model cannot approximate an SDF properly when feeded "perfect" input data.
A small randomized offset suffices in order properly approximate the given SDf from data.

Observed via:

`python .\src\generators\sphere.py --setting distance --radius 1.0 --datapoints 100 --delta 0.0` --> Leads to the inability to approximate.

`python .\src\generators\sphere.py --setting distance --radius 1.0 --datapoints 100 --delta 0.1` --> Proper approximation possible.

Both scenarios get called via:

`python .\application.py --autorun .\data\generated\2DSDF_Circle.json --no-batch --autosave`
