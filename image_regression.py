import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.datasets import mnist, cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import mean_squared_error, r2_score
from differences_images import LingerImageRegressor

# Load the MNIST dataset
# Load MNIST dataset
mnist = load_digits()

X, y = mnist.data, mnist.target

# Convert labels to pixel values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LingerImageClassifier
regressor = LingerImageRegressor()

# Fit the regressor
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate accuracy
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate precision
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Calculate recall
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)