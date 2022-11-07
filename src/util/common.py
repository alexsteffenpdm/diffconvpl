from typing import Dict,Any
import random

def rand_color():
    r = lambda: random.randint(0,255)
    return '#{:02x}{:02x}{:02x}'.format(r(), r(), r())

def make_signs(positive: int, negative: int):
    assert positive >=0
    assert negative >=0
    assert (abs(positive)+ abs(negative)) > 0
 
    return ([1.0] * positive) + ([-1.0] * negative)

def build_log_dict(tqdm_dict:Dict[str,Any],loss:float,func:str,positive:int,negative:int,success:bool):
  
    return {
        "Function": func,
        "Iterations": str(tqdm_dict["total"]),
        "Positive": str(positive),
        "Negative": str(negative),
        "Error": str(loss),
        "Duration": str(tqdm_dict["elapsed"]),
        "Iters per Second": str((tqdm_dict["total"])/tqdm_dict["elapsed"]),
        "Success": success,
    }
