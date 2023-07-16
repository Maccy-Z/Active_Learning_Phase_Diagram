import numpy as np
import timeit

RNG = np.random.default_rng(seed=np.random.SFC64())
mean = np.ones(361).astype(np.float32)
cov = np.eye(361).astype(np.float32)

st = timeit.default_timer()
for _ in range(10):
    x = RNG.multivariate_normal(mean, cov, 1000, method='cholesky').T

end = timeit.default_timer()

print(end - st)
print(x.dtype)

# default: 4.44
# SFC64: 4.36
# MT19937: 4.42
# PCG64DXSM: 4.42
# PCG64: 4.36