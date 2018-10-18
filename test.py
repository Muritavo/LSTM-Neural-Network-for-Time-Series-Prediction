import time
import sys, json
import numpy as np
array_de_arrays = []
for i in range(10):
    array_de_arrays.append([[0,1,2,3,4], [0,2,4,6,8]])
array_de_arrays = np.array(array_de_arrays).astype(float)

for window in array_de_arrays:
    print(window.shape[1])