# Diffconvpl
## General
***

Version: 0.1.1


### Installation of requirements
In order to be able to run the source code the required packages need to be installed.
This can be done via pip with the following command:

`pip install -r requirements_dev.txt`

In order to run the tests please install the required packages via

`pip install -r requirements_test.txt`

### Running the source code

Locate the `application.py` and simply run it with Python. This will run with the following default arguments

        APP_ARGS = {
            "func": "torch.sin(10*x)",
            "params": ["x"],
            "m": 32,
            "entries": 100000,
            "epochs": 5000,
            "positive_funcs": 16,
            "negative_funcs": 16,
            "fullplot": fullplot,
        }


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
Located under **`src\generators`** the scripts will produce the aforementioned `JSON`-Files under **`data\generated`** .

### Example call

`python .\application.py --autorun data\generated\2DSDF_Rectangle_BETA.json --no-batch --autosave`

### Generators and Commandline Arguments

The following generators exist and provide the functionality to parse specific commandline arguments as shown below:

*Note: The square generator can only handle datapoints in increments of 4.*
*Note: The triangle generator can only handle datapoints in increments of 3.*

**Generator-Source: sphere.py**
```
usage: sphere.py [-h] [--setting [<class 'str'>]] [--radius [<class 'float'>]]
                 [--datapoints [<class 'int'>]] [--delta [<class 'float'>]]

SDF Data Generator - Sphere

optional arguments:
  -h, --help            show this help message and exit
  --setting [<class 'str'>]
                        Compute SDF sample data (distance or normals)
  --radius [<class 'float'>]
                        Set the radius for the generated sphere surface.
  --datapoints [<class 'int'>]
                        Set the amount of datapoints generated.
  --delta [<class 'float'>]
                        Randomly varies the range of distances for generated
                        points. Only applicable with setting 'distance'.
```

**Generator-Source: square.py**
```
usage: square.py [-h] [--setting [<class 'str'>]] [--width [<class 'float'>]]
                 [--height [<class 'float'>]] [--datapoints [<class 'int'>]]

SDF Data Generator - Rectangle

optional arguments:
  -h, --help            show this help message and exit
  --setting [<class 'str'>]
                        Compute SDF sample data (distance or normals)
  --width [<class 'float'>]
                        Set width of the rectangle.
  --height [<class 'float'>]
                        Set height of the rectangle.
  --datapoints [<class 'int'>]
                        Set the amount of datapoints generated.
```

**Generator-Source: triangle.py**
```
usage: triangle.py [-h] [--setting [<class 'str'>]]
                   [--radius [<class 'float'>]]
                   [--corners [<built-in function array>]]
                   [--datapoints [<class 'int'>]]

SDF Data Generator - Triangle

optional arguments:
  -h, --help            show this help message and exit
  --setting [<class 'str'>]
                        Compute SDF sample data (distance or normals)
  --radius [<class 'float'>]
                        Defines the radius of the circle, on which the corner
                        points of the triangle reside. (Exclusive vs 'corners'
                        option)
  --corners [<built-in function array>]
                        Defines the coordinates of the corner points. Format:
                        x1,y2,x2,y2,x3,y3 (Exclusive vs 'radius' option)
  --datapoints [<class 'int'>]
                        Set the amount of datapoints generated.
```




