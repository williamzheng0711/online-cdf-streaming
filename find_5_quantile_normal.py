import numpy as np
import matplotlib.pyplot as pyplot



estimated_values = []
Ms = np.arange(start=10, stop=50000, step=10)


for M in Ms:
    mu = 0
    sigma = 1
    samples = np.random.normal(mu, sigma, M)
    estimated = np.quantile(a = samples, q=0.05)
    estimated_values.append(estimated)

pyplot.plot(Ms, estimated_values)
pyplot.axhline(-1.6448, color="red")
pyplot.show()

