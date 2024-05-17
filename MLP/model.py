import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import math

np.random.seed(0)
x = np.linspace(-10, 15, 500).reshape(-1,1)
y =  -x**3 + 10*x**2 + 2 * x + 1 + np.random.normal(20,200,x.shape)
y = y.ravel()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(800,1200,800), 
                   activation='relu',
                   solver='adam',
                   max_iter=1,
                   warm_start=True, 
                   random_state=42,
                   learning_rate='constant', learning_rate_init=0.001)

# List to store predictions at each epoch
predictions_per_epoch = []

# Custom training loop
n_epochs = 100
for epoch in range(n_epochs):
    mlp.partial_fit(X_train, y_train)
    y_pred_all = mlp.predict(x)
    predictions_per_epoch.append(y_pred_all)

# Plotting the animation
fig, ax = plt.subplots()
line, = ax.plot(x, predictions_per_epoch[0], color='red', label='MLP Predictions')
scatter = ax.scatter(X_test, y_test, color='green', label='Actual Test Data')
ax.legend()
ax.set_title('MLP Regression')
ax.set_xlabel('Input Feature')
ax.set_ylabel('Target')

def update(epoch):
    line.set_ydata(predictions_per_epoch[epoch])
    ax.set_title(f'MLP Regression - Epoch {epoch + 1}')
    return line,

ani = FuncAnimation(fig, update, frames=n_epochs, blit=True, repeat=True)
plt.show()

