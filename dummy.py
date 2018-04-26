import numpy as np


a = [(1, 2), (3, 4)]

b, c = np.average(a, axis=0).flatten()

print(b, c)
