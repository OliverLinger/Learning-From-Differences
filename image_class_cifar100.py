from keras.datasets import cifar100
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from differences_images import LingerImageClassifier

# Load CIFAR-100 dataset
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar100.load_data()

# Reshape data
X_train_cifar = X_train_cifar.reshape(X_train_cifar.shape[0], -1)
X_test_cifar = X_test_cifar.reshape(X_test_cifar.shape[0], -1)

# Initialize LingerImageClassifier
classifier_cifar = LingerImageClassifier()

# Fit the classifier on CIFAR-100
classifier_cifar.fit(X_train_cifar, y_train_cifar)

# Predict on the CIFAR-100 test set
y_pred_cifar = classifier_cifar.predict(X_test_cifar)

# Evaluate performance on CIFAR-100
accuracy_cifar = accuracy_score(y_test_cifar, y_pred_cifar)
precision_cifar = precision_score(y_test_cifar, y_pred_cifar, average='weighted')
recall_cifar = recall_score(y_test_cifar, y_pred_cifar, average='weighted')
f1_cifar = f1_score(y_test_cifar, y_pred_cifar, average='weighted')

print("Accuracy (CIFAR-100):", accuracy_cifar)
print("Precision (CIFAR-100):", precision_cifar)
print("Recall (CIFAR-100):", recall_cifar)
print("F1-score (CIFAR-100):", f1_cifar)
