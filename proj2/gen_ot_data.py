import numpy as np
from time import time
from imageio.v2 import imread
from numpy.linalg import norm
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

# Read source image
source = imread('source.png')
print(f'Source image size: {source.shape[0]} x {source.shape[1]}.')

# Read destination image
dest = imread('dest.png')
print(f'Dest image size: {dest.shape[0]} x {dest.shape[1]}.')

# Reshape the source and dest into vector and normalize so that they both
# sum to 1
source = source.reshape(-1)
dest = dest.reshape(-1)
source = source.astype(float) / sum(source)
dest = dest.astype(float) / sum(dest)

# Make sure source and dest are valid probability distributions
assert np.all(source >= 0)
assert np.all(dest >= 0)
assert np.abs(sum(source) - 1) < 1e-08 and np.abs(sum(dest) - 1) < 1e-08
print('Source and destination distributions generated.')

# Generate the squared Euclidean cost function C. The cost efficient
# C_{i, j} between position i = (ix,iy) at the original source image and
# position j = (jx, jy) at the original destination image is defined as
# C_{i, j} = (ix - jx)^2 + (iy - jy)^2.
# This code assumes that both images are square and have the same size.
# Please modify the code accordingly if the data does not meet this
# assumption.
sz = imread('source.png').shape[0]
ii = np.tile(np.arange(0, sz), sz).reshape(-1, 1)
jj = np.repeat(np.arange(0, sz), sz).reshape(-1, 1)
C = (ii - ii.T) ** 2 + (jj - jj.T) ** 2
print('Cost function generated.')

# Solve the optimal transportation problem (as a special case of LP) using
# scipy's linprog.
c = C.reshape(-1).copy()
m = n = sz ** 2
i_pos = np.hstack([np.repeat(np.arange(0, m), n), np.tile(np.arange(m, n + m), m)])
j_pos = np.hstack([np.arange(0, m * n), np.arange(0, m * n)])
A_eq = csr_matrix((np.ones(2 * m * n), (i_pos, j_pos)), shape=(m + n, m * n))
b_eq = np.concatenate([source, dest])

# by default lb = 0 and ub = None unless specified with bounds.
# method can be ‘highs’ (default), ‘highs-ds’, ‘highs-ipm’
# tol 1e-6 useless, no iter display
print('\nSolving the LP using linprog ...')
t0 = time()
res = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs-ipm', options={'disp': True})
t1 = time() - t0

x = res.x
print(res.message)
print(f'linprog time: {t1:.1f} seconds.')
print(f'Optimal transportation cost: {c.T @ x:.8e}.')
print(f'Dual objective value: {b_eq.T @ res.eqlin.marginals:.8e}.')
print(f'Relative duality gap: {np.abs(c.T @ x + b_eq.T @ res.eqlin.marginals) / (1 + 0.5 * np.abs(c.T @ x) - 0.5 * (b_eq.T @ res.eqlin.marginals)):.2e}')
print(f'Relative primal infeasibility: {norm(A_eq @ x - b_eq, np.inf) / (1 + norm(b_eq, np.inf)):.2e}.')
print(f'Relative dual infeasibility: {norm(A_eq.T @ -res.eqlin.marginals + res.lower.marginals - c, np.inf) / (1 + norm(c, np.inf)):.2e}.')
