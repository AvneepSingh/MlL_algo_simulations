import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(0)
x = np.linspace(-5, 5, 300)
y =  -x**3 + 10*x**2 + 2 * x + 1 + np.random.normal(10, 70,x.shape)

def polynomial(x, coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

fig, ax = plt.subplots()
ax.scatter(x, y, label='Data')
line, = ax.plot([], [], 'r-', label='Fit')
ax.set_xlim(-5, 5)
ax.set_ylim(min(y)-10, max(y)+10)
ax.legend()

degree = 35
coeffs_list = []

def fit_polynomial(degree, x, y):
    for deg in range(1, degree + 1):
        coeffs = np.polyfit(x, y, deg)
        coeffs_list.append(coeffs)

# Fit the polynomial and store coefficients for each degree
fit_polynomial(degree, x, y)
50
# Initialization function for animation
def init():
    line.set_data([], [])
    return line,

# Animation function
def animate(i):
    coeffs = coeffs_list[i]
    y_fit = polynomial(x, coeffs[::-1])
    line.set_data(x, y_fit)
    ax.set_title(f'Polynomial Degree: {i + 1}')
    return line,

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(coeffs_list), interval=500, blit=True)

plt.show()
