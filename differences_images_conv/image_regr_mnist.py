from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import ssl

from sklearn.metrics import mean_squared_error
import _regression

# Adjust SSL settings for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

# Import your custom regressor

# Load MNIST dataset
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

# Initialize the LingerImageRegressor with parameters suitable for your regression task
regressor_mnist = _regression.LingerImageRegressor(n_neighbours_1=2, n_neighbours_2=2)

# Convert labels to float32 for regression and reshape
y_train_mnist = y_train_mnist.astype('float32').reshape(-1, 1)
y_test_mnist = y_test_mnist.astype('float32').reshape(-1, 1)

# Reshape MNIST images for the regressor and CNN input
X_train_mnist_reshaped = X_train_mnist.reshape(X_train_mnist.shape[0], 28, 28, 1)
X_test_mnist_reshaped = X_test_mnist.reshape(X_test_mnist.shape[0], 28, 28, 1)

# Fit the regressor on the training data
diff_X, diff_y = regressor_mnist.fit(X_train_mnist_reshaped, y_train_mnist)

# Convert differences to numpy arrays
diff_X = np.array(diff_X).reshape(-1, 28, 28, 1)
diff_y = np.array(diff_y)

# Define the CNN model for regression
def create_regression_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Output layer for regression
    ])
    return model

# Define a function to create a CNN model for regression
def create_standard_regression_cnn(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Output layer for regression
    ])
    return model

# Initialize the standard CNN model
standard_model = create_standard_regression_cnn((28, 28, 1))

# Compile the standard model
standard_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the standard model on the original MNIST dataset
standard_model.fit(X_train_mnist_reshaped, y_train_mnist, epochs=50, batch_size=32, validation_split=0.2)


# Initialize the CNN model
model = create_regression_cnn((28, 28, 1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

diff_y = diff_y.reshape(-1)
# Train the model on the difference dataset
model.fit(diff_X, diff_y, epochs=50, batch_size=32, validation_split=0.2)

# Use the trained CNN model to make predictions on test data
y_pred = regressor_mnist.predict(X_test_mnist, model=model,input_shape=(28, 28, 1))

mnse_mnist = mean_squared_error(y_test_mnist, y_pred)
print("Mean Squared Error MNIST:", mnse_mnist)

# Calculate Variance of y_test_mnist to use in MNSE calculation
variance = np.var(y_test_mnist)


print("Mean Normalized Squared Error MNIST:", mnse_mnist)

y_pred_standard = standard_model.predict(X_test_mnist_reshaped)

# Calculate MSE for the standard model
mnse_standard = mean_squared_error(y_test_mnist, y_pred_standard)
print("Standard CNN Mean Squared Error:", mnse_standard)

# Calculate MNSE for the standard model
print("Standard CNN Mean Normalized Squared Error:", mnse_standard)

file_path = r'C:\Users\USER\final_year\fyp\results\images\mnist.txt'
with open(file_path, 'a') as file:
    file.write(f"Accuracy on MNIST test set differences: {mnse_mnist}\n")
    file.write(f"Standard CNN Accuracy on MNIST test set: {mnse_standard}\n")