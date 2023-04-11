import inspect
from multiprocessing import Process
from multiprocessing import shared_memory
from typing import Callable, Dict, Any
import time
import numpy as np
import json


class MemorySharedSubprocess:
    def __init__(self, target: Callable):
        self.shm: shared_memory.SharedMemory = None
        self.target: Callable = target
        self.forked: bool = False
        self.subprocess: Process = None
        self.targetargs = inspect.signature(self.target)

    def _create_buffer(self, data: Dict[str, Any]):
        d = bytes(json.dumps(data).encode("UTF-8"))
        self.shm = shared_memory.SharedMemory(create=True, size=len(d))
        self.shm.buf[0 : len(d)] = d

    def _unwrap(
        self,
    ):
        # print(
        #     f"Unwrapping shm with name {self.shm.name} to target function {self.target.__name__}"
        # )
        data = json.loads(self.shm.buf.tobytes().decode("UTF-8").rstrip("\x00"))
        call_str = f"self.target("
        for t, k in zip(self.targetargs.parameters, data.keys()):
            assert t == k, f"Key '{k}' does not match expected parameter '{t}'"

        for k, v in data.items():
            if type(v) == list:
                data[k] = np.asanyarray(v)
            call_str += f'data["{k}"], '

        call_str = call_str[:-2] + ")"
        eval(call_str)

    def fork_on(self, data: Dict[str, Any]):
        self._create_buffer(data)
        if self.shm.size == 0:
            raise ValueError(
                "Cannot fork. Shared memory buffer has not been created until now."
            )

        self.process = Process(target=self._unwrap)
        self.forked = True
        self.process.start()

    def await_join(
        self,
    ):
        if self.forked == False:
            raise ValueError("Cannot join. The subprocess has not been forked yet.")

        while self.process.is_alive():
            time.sleep(1)
        try:
            self.process.join()
        finally:
            self.shm.close()
            self.shm.unlink()
            self.forked = False
