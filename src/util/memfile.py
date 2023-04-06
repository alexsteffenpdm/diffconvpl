from shm_process import MemorySharedSubprocess
from plot import plotsdf
import random
from typing import Dict, Any
import json


def test_conversion(data: Dict[str, Any]):
    j = bytes(json.dumps(data).encode("UTF-8"))
    return json.loads(bytes(j))


if __name__ == "__main__":
    func = "np.sqrt(x**2 + y**2) - 0.25"
    s = 100
    xv = [round(random.random(), 2) for _ in range(s)]
    yv = [round(random.random(), 2) for _ in range(s)]
    z = [[round(random.random(), 2) for _ in range(s)] for _ in range(s)]
    err_d = [round(random.random(), 2) for _ in range(s)]
    err_v = [round(random.random(), 2) for _ in range(s)]
    autosave = False
    losses = [round(random.random(), 2) for _ in range(s)]
    filename = "memfile_test"

    args: Dict[str, Any] = {
        "func": func,
        "xv": xv,
        "yv": yv,
        "z": z,
        "err_d": err_d,
        "err_v": err_v,
        "autosave": autosave,
        "losses": losses,
        "filename": filename,
    }

    # args = test_conversion(args)
    # for k,v in args.items():
    #     print(k,v)

    mshp: MemorySharedSubprocess = MemorySharedSubprocess(target=plotsdf)
    mshp.create_buffer(data=args)
    mshp.fork()
    mshp.join()
