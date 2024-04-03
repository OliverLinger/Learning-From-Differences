from differences_images_conv import _classification
from keras.datasets import cifar100
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load CIFAR-100 dataset
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar100.load_data()

# Initialize the DifferenceImageGenerator
Linger_image_classifier = _classification.LingerImageClassifier(n_neighbours_1=5, n_neighbours_2=5)

# Fit the k-nearest neighbors model on the dataset
print(X_train_cifar.shape)
print(y_train_cifar.shape)
quit()
diff_X, diff_y = Linger_image_classifier.fit(X_train_cifar, y_train_cifar)
diff_X = np.array(diff_X)
diff_y = np.array(diff_y)
diff_X = diff_X.reshape(-1, 32, 32, 3)  # Assuming RGB 
# Define a simple CNN model
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

# Example usage:
input_shape = (32, 32, 3)  # Input shape for CIFAR-100
num_classes = 100  # Number of classes in CIFAR-100
diff_y = np.zeros((diff_y.shape[0], num_classes))
# Create and compile the CNN model
cnn_model = create_simple_cnn(input_shape, num_classes)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model on the differences_X and differences_y
cnn_model.fit(diff_X, diff_y, epochs=1, batch_size=32, validation_split=0.2)

# Use the trained CNN model to make predictions on test data
y_pred = Linger_image_classifier.predict(X_test_cifar, input_shape=(32, 32, 3), model=cnn_model)

# Calculate accuracy
accuracy_cifar_100 = accuracy_score(y_test_cifar, y_pred)
print("Accuracy on CIFAR-100 test set:", accuracy_cifar_100)