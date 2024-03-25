from keras.datasets import mnist
import numpy as np
from differences_images_conv import LingerImageRegressor  # Import the regressor
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load MNIST dataset
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

# Initialize LingerImageRegressor
regressor_mnist = LingerImageRegressor(n_neighbours_1=5)

# Fit the regressor on MNIST
differences_X, differences_y = regressor_mnist.fit(X_train_mnist, y_train_mnist)
differences_X = np.array(differences_X)

# Reshape input data for CNN
differences_X = differences_X.reshape(-1, 28, 28, 1)  # Assuming grayscale images

# Define the CNN architecture for regression
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Output layer with linear activation for regression

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Use mean squared error loss for regression
              metrics=['mae'])  # Use mean absolute error as metric

# Train the model
model.fit(differences_X, differences_y, epochs=3, batch_size=32, validation_split=0.2)

# Use the trained CNN model to make predictions on test data
predictions = regressor_mnist.predict(X_test_mnist, model=model)

# Perform further evaluation for regression, such as calculating RMSE or plotting actual vs. predicted values.
