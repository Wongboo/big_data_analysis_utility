# Implementation of the Wirtinger Flow (WF) algorithm presented in the paper
# "Phase Retrieval via Wirtinger Flow: Theory and Algorithms"
# by E. J. Candes, X. Li, and M. Soltanolkotabi

# The input data are coded diffraction patterns about a random complex
# valued 1D signal.


# Make signal
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


n = 128
x = np.random.randn(n, 1) + 1j * np.random.randn(n, 1)

# Make masks and linear sampling operators
L = 6

# Sample phases: each symbol in alphabet {1, -1, i , -i} has equal prob.
Masks = np.random.choice([1j, - 1j, 1, - 1], (n, L))
# Sample magnitudes and make masks
temp = np.random.rand(*Masks.shape)
Masks = Masks * ((temp <= 0.2) * np.sqrt(3) + (temp > 0.2) / np.sqrt(2))
# Make linear operators; A is forward map and At its scaled adjoint (At(Y)*numel(Y) is the adjoint)
def A(I): return np.fft.fft(Masks.conj() * np.tile(I, (1, L)))
def At(Y): return np.mean(Masks * np.fft.ifft(Y), axis=1)


# Data
Y = np.abs(A(x)) ** 2

# Initialization
npower_iter = 50
z0 = np.random.randn(n, 1)
z0 = z0 / norm(z0)
for tt in np.arange(npower_iter):
    z0 = At(Y * A(z0)).reshape(-1, 1)
    z0 = z0 / norm(z0)

normest = np.sqrt(np.sum(Y) / np.asarray(Y).size)
z = normest * z0
Relerrs = [norm(x - np.exp(- 1j * np.angle(np.trace(x.T @ z))) * z) / norm(x)]

# Loop
T = 2500
tau0 = 330
def mu(t): return min(1 - np.exp(- t / tau0), 0.2)
# def mu(t=None): return np.amin(1 - np.exp(- t / tau0), 0.2)


for t in np.arange(T):
    Bz = A(z)
    C = (np.abs(Bz) ** 2 - Y) * Bz
    grad = At(C)
    z = z - (mu(t + 1) / normest ** 2 * grad).reshape(-1, 1)
    Relerrs.append(norm(x - np.exp(-1j * np.angle(np.trace(x.T @ z))) * z) / norm(x))

# Check results
print(f'Relative error after initialization: {Relerrs[0]}')
print(f'Relative error after {T} iterations: {Relerrs[T]}')
fig, ax = plt.semilogy(np.arange(0, T+1), Relerrs)
plt.xlabel('Iteration')
plt.ylabel('Relative error (log10)')
plt.title('Relative error vs. iteration count')
fig.show()
