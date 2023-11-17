import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
m = 2048
n = 512
p = 20
A = rng.standard_normal((m, p)) @ rng.standard_normal((p, n))

A1 = plt.imread('example.jpg')  # read the image peppers.png
plt.imshow(A1)  # display the image
A = A1[..., :3] @ [0.2989, 0.5870, 0.1140]  # Convert to grayscale and double [0.2125, 0.7154, 0.0721]
