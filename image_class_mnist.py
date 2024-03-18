from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from keras.datasets import mnist, cifar10
from sklearn.model_selection import train_test_split
from differences_images import LingerImageClassifier

print("mnist")
# Load MNIST dataset
mnist = load_digits()

X, y = mnist.data, mnist.target

# Split data into train and test sets
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LingerImageClassifier
classifier_mnist = LingerImageClassifier()

# Fit the classifier on MNIST
classifier_mnist.fit(X_train_mnist, y_train_mnist)

# Predict on the MNIST test set
y_pred_mnist = classifier_mnist.predict(X_test_mnist)

# Evaluate performance on MNIST
accuracy_mnist = accuracy_score(y_test_mnist, y_pred_mnist)
precision_mnist = precision_score(y_test_mnist, y_pred_mnist, average='weighted')
recall_mnist = recall_score(y_test_mnist, y_pred_mnist, average='weighted')
f1_mnist = f1_score(y_test_mnist, y_pred_mnist, average='weighted')

print("Accuracy (MNIST):", accuracy_mnist)
print("Precision (MNIST):", precision_mnist)
print("Recall (MNIST):", recall_mnist)
print("F1-score (MNIST):", f1_mnist)
