# function Test_MC

# min mu*||X||_* + sum_{ij in idx}

import numpy as np
from numpy import round, floor, sqrt


# generate data
m = 40; n = 40; sr = 0.3; p = int(round(m*n*sr)); r = 3  # noqa: E702
# m = 100; n = m; sr = 0.3; p = int(round(m*n*sr)); r = 2  # noqa: E702
# m = 100; n = m ; p = 5666; sr = p/m/n; r = 10  # noqa: E702
# m = 200; n = m; p = 15665; sr = p/m/n; r = 10  # noqa: E702
# m = 500; n = m; p = 49471; sr = p/m/n; r = 10  # noqa: E702
# m = 150; n = 300; sr = 0.49; p = int(round(m*n*sr)); r = 10  # noqa: E702


# fr is the freedom of set of rank-r matrix, maxr is the maximum rank one
# can recover with p samples, which is the max rank to keep fr < 1
fr = r*(m+n-r)/p
maxr = floor(((m+n)-sqrt((m+n)**2-4*p))/2)

rs = 2021
rng = np.random.default_rng(rs)

# get problem
omega = rng.permutation(m*n)[:p]  # Omega gives the position of samplings
xl = rng.standard_normal((m, r))
xr = rng.standard_normal((n, r))
a = xl @ xr.T  # A is the matrix to be completed
m_mat = a.reshape(-1)[omega]  # M is the samples from A
