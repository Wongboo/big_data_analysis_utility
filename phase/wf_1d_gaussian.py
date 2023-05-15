# Implementation of the Wirtinger Flow (WF) algorithm presented in the paper
# "Phase Retrieval via Wirtinger Flow: Theory and Algorithms"
# by E. J. Candes, X. Li, and M. Soltanolkotabi

# The input data are phaseless measurements about a random complex
# valued 1D signal.

# Make signal and data
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


n = 128
x = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)
m = int(np.round(4.5 * n))
A = 1 / np.sqrt(2) * np.random.randn(m, n) + 1j / np.sqrt(2) * np.random.randn(m, n)
y = np.abs(A @ x) ** 2

# Initialization
npower_iter = 50
z0 = np.random.randn(n, 1)
z0 = z0 / norm(z0)
for tt in np.arange(npower_iter):
    z0 = A.T @ (y * (A @ z0))
    z0 = z0 / norm(z0)

normest = np.sqrt(sum(y) / np.asarray(y).size)
z = normest * z0
Relerrs = [norm(x - np.exp(- 1j * np.angle(np.trace(x.T @ z))) * z) / norm(x)]

# Loop
T = 2500
tau0 = 330
def mu(t): return min(1 - np.exp(- t / tau0), 0.2)


for t in np.arange(T):
    yz = A @ z
    grad = 1 / m * A.T @ ((np.abs(yz) ** 2 - y) * yz)
    z = z - mu(t + 1) / normest ** 2 * grad
    Relerrs.append(norm(x - np.exp(-1j * np.angle(np.trace(x.T @ z))) * z) / norm(x))

# Check results
print(f'Relative error after initialization: {Relerrs[0]}')
print(f'Relative error after {T} iterations: {Relerrs[T]}')
fig, ax = plt.semilogy(np.arange(0, T+1), Relerrs)
plt.xlabel('Iteration')
plt.ylabel('Relative error (log10)')
plt.title('Relative error vs. iteration count')
fig.show()