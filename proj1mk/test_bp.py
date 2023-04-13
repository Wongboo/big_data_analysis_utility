# function Test_BP

# min mu*||x||_1 + ||Ax-b||_1

import time
import numpy as np
import scipy.sparse as sp
from numpy.random import rand, randn
from numpy.linalg import norm


def sprandn(m, n, density):
    return sp.random(m, n, density=density, format='csc',
                     data_rvs=lambda s: np.random.normal(size=s))


# generate data
n = 1024
m = 512
A = randn(m, n)
u = sprandn(n, 1, 0.1)
b = A*u
mu = 1e-2
x0 = rand(n, 1)


def errfun(x1, x2): return norm(x1-x2)/(1+norm(x1))
def resfun(x): return norm(A@x-b, 1)
def nrm1fun(x): return norm(x, 1)


# cvxpy calling mosek
opts1 = {}  # modify options
start_time = time.time()
x1, out1 = l1_cvx_mosek(x0, A, b, mu, opts1)
t1 = time.time() - start_time

# direct calling gurobi
opts2 = opts_gurobi()  # modify options
start_time = time.time()
x2, out2 = l1_gurobi(x0, A, b, mu, opts2)
t2 = time.time() - start_time

# admm
opts5 = opts_admm()  # modify options
start_time = time.time()
x5, out5 = l1_admm(x0, A, b, mu, opts5)
t5 = time.time() - start_time


# print comparison results with cvx-call-mosek
print(f'cvxpy-mosek:  res: {resfun(x1):3.2e}, nrm1: {nrm1fun(x1):3.2e}, cpu: {t1:5.2f}')    # noqa:E501
print(f'gurobi:       res: {resfun(x2):3.2e}, nrm1: {nrm1fun(x2):3.2e}, cpu: {t2:5.2f}, err-to-cvxpy-mosek: {errfun(x1, x2):3.2e}')  # noqa:E501
print(f'admm:         res: {resfun(x5):3.2e}, nrm1: {nrm1fun(x5):3.2e}, cpu: {t5:5.2f}, err-to-cvxpy-mosek: {errfun(x1, x5):3.2e}')  # noqa:E501
