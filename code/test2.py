import numpy as np
from matplotlib import pyplot as plt

x1 = np.array([[1, 2], [3, 4], [5, 6]])
x2 = np.array([[1, 2], [3, 1], [5, 4]])

row_true = np.isin(x1, x2)

print(row_true)
print()
print(np.all(row_true, axis=1))




