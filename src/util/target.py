from typing import List, Optional
import torch
import numpy as np

class Target(object):

        
    class UnknownPackageError(Exception):
        pass

    class UnspecifiedPackageError(Exception):
        pass


    def __init__(self,func: str,parameters:List[str]):
        self.func: str= func
        self.parameters:List[str] = parameters
        self.contains_package: bool = True if "np." in self.func or "torch." in self.func else False


    def __repr__(self):
        return f"lambda {','.join(self.parameters)}: {self.func}"

    def no_package_str(self):
        if self.contains_package:
            return self.func.split(".",1)[1]
        else:
            return self.func


    def as_lambda(self,package: Optional[str]=None):
        if not self.contains_package:
            return eval(f"lambda {','.join(self.parameters)}: {self.func}")    
        else:
            if package is not None:
                if package == "numpy":
                    return eval(f"lambda {','.join(self.parameters)}: {self.func.replace('torch','np') if 'torch' in self.func else self.func}")    
                elif package == "torch":
                    return eval(f"lambda {','.join(self.parameters)}: {self.func.replace('np','torch') if 'np' in self.func else self.func}")
                else:
                    raise self.UnknownPackageError("Could not identify package used in Lambda function evaluation")
            else:
                raise self.UnspecifiedPackageError(f"Target function '{self.func}' requires a package interpretation to be defined please use as_lambda (package=<package_name>) ")
