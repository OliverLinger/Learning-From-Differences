from collections import Counter
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LingerImageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_neighbours_1=5,
        n_neighbours_2=5,
    ):
        """
        Initialize the DifferenceImageGenerator.

        Parameters:
        - n_neighbours: Number of nearest neighbors to consider for computing difference images.
        """
        
        self.n_neighbours_1 = n_neighbours_1
        self.n_neighbours_2 = n_neighbours_2
        self.train_X = []
        self.train_y = []


    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples, height, width, channels)
            Training images.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        differences_X : list
            List of arrays representing differences between nearest neighbor images.
        differences_y : list
            Target differences.
        """
        # Reshape X if necessary
        if len(X.shape) == 4:
            X = X.reshape(len(X), -1)
        else:
            X = X
        self.n_neighbours_1 += 1  # Increment n_neighbours

        # Fit nearest neighbors to find the indices of nearest neighbors
        neighbours = NearestNeighbors(n_neighbors=self.n_neighbours_1).fit(X.reshape(len(X), -1))
        _, indices = neighbours.kneighbors(X.reshape(len(X), -1))


        differences_X = []
        differences_y = []
        for i, indice in enumerate(indices):
            base_image = indice[0]
            neighbors = indice[1:]
            for neighbor_index in neighbors:
                neighbor_image = X[neighbor_index]
                neighbor_target = y[neighbor_index]

                # Calculate the absolute difference between images
                diff = np.abs(X[base_image] - neighbor_image)
                differences_X.append(diff)
                differences_y.append(y[base_image] - neighbor_target)

        self.train_X = X
        self.train_y = y
        self.classes_ = np.unique(y)

        return differences_X, differences_y

    def predict(self, X, model, input_shape):
        """
        Predict target values for the input data using non-weighted k-nearest neighbors (KNN).

        Parameters:
        - X: Input data for prediction.

        Returns:
        - y_pred: Predicted target values.
        """
        # Fit nearest neighbors to the training data
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbours_2).fit(self.train_X.reshape(len(self.train_X), -1))
        _, indices = nbrs.kneighbors(X.reshape(len(X), -1))

        differences_test_X = [
        np.abs(X[test].flatten() - self.train_X[indices[test][nn_in_training]].flatten())
        for test in range(len(X))
        for nn_in_training in range(len(indices[0]))
        ]
        diff_X = np.array(differences_test_X)
        differences_test_X = diff_X.reshape(-1, *input_shape)
        # Make a prediction based on the differences in the test set X
        predictions = model.predict(differences_test_X)
        
        y_pred = []
        for indexes, differences in zip(indices, predictions):
            results = [self.train_y[i][0] + d for i, d in zip(indexes, differences)]
            counts = Counter(results)

            # Retrieve the most common item
            most_common_item = counts.most_common(1)
            if most_common_item:
                y_pred.append(most_common_item[0][0])
            else:
                # Handle the case where there are no items
                y_pred.append(None)
        return y_pred
            
    def get_params(self, deep=True):
        """
        Get parameters for the Linger Classifier.

        Parameters:
        - deep: If True, return parameters for the internal MLPClassifier.

        Returns:
        - params: Dictionary of parameters.

        """
        return {
            "n_neighbours_1": self.n_neighbours_1,
            "n_neighbours_2": self.n_neighbours_2,
            
        }

    def set_params(self, **params):
        """
        Set parameters for the Linger Classifier.

        Parameters:
        - params: Dictionary of parameters.

        Returns:
        - self: The modified Linger Classifier.

        """
        # Set parameters for your custom classifier
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
                # Check if the parameter also exists in the classifier
                if hasattr(self.classifier, param):
                    setattr(self.classifier, param, value)
            else:
                # If the parameter is not part of your custom classifier, set it in the internal MLPClassifier
                if hasattr(self.classifier, param):
                    setattr(self.classifier, param, value)
        return self
