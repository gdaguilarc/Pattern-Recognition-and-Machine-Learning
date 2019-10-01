import numpy as np
import matplotlib.pyplot as plot

# Generate a sine wave
t = np.arange(0, 10, 0.1);
y = np.sin(t)
plot.plot(t, y)
plot.title('Training data for regression y=f(t)')
plot.xlabel('Time')
plot.ylabel('y = sin(t)')
plot.grid(True, which='both')
plot.show()

# Load necessary packages for neural networks
from keras.models import Sequential
from keras.layers import Dense

# Model sequential
model = Sequential()
# First hidden layer (we also need to tell the input dimension)
model.add(Dense(10, input_dim=1, activation='sigmoid'))
# First hidden layer (we also need to tell the input dimension)
model.add(Dense(10, activation='sigmoid'))

# Output layer
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

model.fit(t, y, epochs=5000, verbose=1)

from sklearn.metrics import mean_squared_error 
y_pred = model.predict(t)
print(y[1])
print(y_pred[1])
print(np.sum(np.absolute(np.subtract(y,y_pred)))/len(t))
print(np.square(np.subtract(y,y_pred)).mean())
print(len(y))
print(np.divide(np.sum(np.square(y-y_pred)),len(y)))
print('MSE=',mean_squared_error(y,y_pred))
plot.plot(t, y, label='y')
plot.plot(t, y_pred, label='y_pred')
plot.title('Training data (sive wave)')
plot.xlabel('Time')
plot.ylabel('y = sin(t)')
plot.grid(True, which='both')
plot.legend()
plot.show()