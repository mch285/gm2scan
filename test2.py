import os
import pathlib
import numpy as np
path = str(pathlib.Path(os.path.abspath(__file__)).parents[1]) + f"/gm2mt/results"


arr = np.arange(1, 13).reshape((3, 4))
print(arr)
print(arr[(1, 2)])