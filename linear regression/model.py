import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation

# Step 1: Generate synthetic data
np.random.seed(0)
x = np.linspace(-5, 5, 100)
y = 10 * x + 1 + np.random.normal(0, 10, x.shape)

# Step 2: Create a linear regression model
model = LinearRegression()

# Step 3: Animate the training process
fig, ax = plt.subplots()
ax.scatter(x, y, label='Data')
line, = ax.plot([], [], 'r-', label='Fit')
ax.set_xlim(-5, 5)
ax.set_ylim(min(y)-10, max(y)+10)
ax.legend()

def init():
    line.set_data([], [])
    return line,

def animate(i):
    # Fit the model using a subset of data points
    model.fit(x[:i+1].reshape(-1, 1), y[:i+1])
    # Predict values for all data points
    y_pred = model.predict(x.reshape(-1, 1))
    line.set_data(x, y_pred)
    ax.set_title(f'Num Points: {i + 1}')
    return line,

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=100, blit=True)

plt.show()
