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
def create_simple_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# One-hot encode CIFAR-100 labels
num_classes = 100
y_train_cifar_one_hot = to_categorical(y_train_cifar, num_classes)
y_test_cifar_one_hot = to_categorical(y_test_cifar, num_classes)

# Initialize LingerImageClassifier and fit to generate difference images
linger_image_classifier = _classification.LingerImageClassifier(n_neighbours_1=5, n_neighbours_2=5)
diff_X, diff_y = linger_image_classifier.fit(X_train_cifar, y_train_cifar)
diff_X = np.array(diff_X).reshape(-1, 32, 32, 3)  # Reshape difference images
diff_y = np.array(diff_y)
# Example usage:
input_shape = (32, 32, 3)  # Input shape for CIFAR-100
num_classes = 100  # Number of classes in CIFAR-100
diff_y = np.zeros((diff_y.shape[0], num_classes))

# Initialize and compile the CNN model for training on difference images
input_shape = (32, 32, 3)
difference_cnn_model = create_simple_cnn(input_shape, num_classes)
difference_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
difference_cnn_model.fit(diff_X, diff_y, epochs=1, batch_size=32, validation_split=0.2)

# Additionally, train a standard CNN model directly on CIFAR-100 images for comparison
standard_cnn_model = create_simple_cnn(input_shape, num_classes)
standard_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
standard_cnn_model.fit(X_train_cifar, y_train_cifar_one_hot, epochs=1, batch_size=32, validation_split=0.2)

# Use the standard CNN model to make predictions on the test data
y_pred_standard = np.argmax(standard_cnn_model.predict(X_test_cifar), axis=1)

# Use the trained CNN model to make predictions on test data
y_pred = linger_image_classifier.predict(X=X_test_cifar, model=difference_cnn_model,input_shape=(32, 32, 3))

# Calculate and print accuracy for the standard CNN model
accuracy_standard_cnn = accuracy_score(y_test_cifar.flatten(), y_pred_standard)
print("Standard CNN Accuracy on CIFAR-100 test set:", accuracy_standard_cnn)

# Calculate accuracy
accuracy_cifar_100 = accuracy_score(y_test_cifar, y_pred)
print("Accuracy on CIFAR-100 test set differences:", accuracy_cifar_100)