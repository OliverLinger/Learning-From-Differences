import numpy as np
import ssl
import _classification
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score

# Ensure secure download of CIFAR-100 dataset
ssl._create_default_https_context = ssl._create_unverified_context

# Load CIFAR-100 dataset
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar100.load_data()

# Function to create a CNN model
def create_simple_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='linear')  # Use linear activation for regression
    ])
    return model



# Initialize LingerImageClassifier and fit to generate difference images
linger_image_classifier = _classification.LingerImageClassifier(n_neighbours_1=5, n_neighbours_2=5)
diff_X, diff_y = linger_image_classifier.fit(X_train_cifar, y_train_cifar)
diff_X = np.array(diff_X).reshape(-1, 32, 32, 3)  # Reshape difference images
diff_y = np.array(diff_y)

# Example usage:
input_shape = (32, 32, 3)
difference_cnn_model = create_simple_cnn(input_shape)
difference_cnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
difference_cnn_model.fit(diff_X, diff_y, epochs=1, batch_size=32, validation_split=0.2)

# Additionally, train a standard CNN model directly on CIFAR-100 images for comparison
standard_cnn_model = create_simple_cnn(input_shape)
standard_cnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
standard_cnn_model.fit(X_train_cifar, y_train_cifar, epochs=1, batch_size=32, validation_split=0.2)

# Use the standard CNN model to make predictions on the test data
y_pred_standard = np.argmax(standard_cnn_model.predict(X_test_cifar), axis=1)

# Use the trained CNN model to make predictions on test data
y_pred = linger_image_classifier.predict(X=X_test_cifar, model=difference_cnn_model,input_shape=(32, 32, 3))
print(y_test_cifar.flatten())
print("-----------------------------------------------------")
y_pred = np.round(y_pred).astype(int)
print(y_pred)
# Calculate and print accuracy for the standard CNN model
accuracy_standard_cnn = accuracy_score(y_test_cifar.flatten(), y_pred_standard)
print("Standard CNN Accuracy on CIFAR-100 test set:", accuracy_standard_cnn)

# Calculate accuracy
accuracy_cifar_100 = accuracy_score(y_test_cifar, y_pred.flatten())
print("Accuracy on CIFAR-100 test set differences:", accuracy_cifar_100)
file_path = r'C:\Users\USER\final_year\fyp\results\images\cifar.txt'
with open(file_path, 'a') as file:
    file.write(f"Accuracy on CIFAR-100 test set differences: {accuracy_cifar_100}\n")
    file.write(f"Standard CNN Accuracy on CIFAR-100 test set: {accuracy_standard_cnn}\n")

