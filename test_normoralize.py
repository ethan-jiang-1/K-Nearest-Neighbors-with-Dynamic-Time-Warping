import numpy as np
from sklearn.preprocessing import normalize

x = np.random.rand(10) *20 - np.random.rand(10) *15
norm1 = x / np.linalg.norm(x)
norm2 = normalize(x[:,np.newaxis], axis=0).ravel()

print(x)
print(norm1)
print(norm2)
print(np.all(norm1 == norm2))
# True