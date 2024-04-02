# Authors: Oliver Linger

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from skimage.metrics import structural_similarity


class LingerImageRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_neighbours_1=5,
        n_neighbours_2=5,

    ):
        """
        Linger Regressor is a custom regressor based on k-nearest neighbors and MLPRegressor that implements classification to solve regression tasks.

        Parameters:
        - n_neighbours_1: Number of neighbors for the first k-nearest neighbors search during fitting.
        - n_neighbours_2: Number of neighbors for the second k-nearest neighbors search during prediction.
        - hidden_layer_sizes: Tuple defining the size of hidden layers in the MLPRegressor.
        - activation: Activation function for MLPRegressor.
        - solver: Solver for MLPRegressor.
        - ... (other parameters for MLPRegressor)

        Attributes:
        - regressor: Internal MLPRegressor used for training and prediction.
        - train_X, train_y: Training data stored after fitting.

        """
        self.n_neighbours_1 = n_neighbours_1
        self.n_neighbours_2 = n_neighbours_2

    def fit(self, X, y):
        if len(X.shape) == 4:
            X = X.reshape(len(X), -1)
        self.n_neighbours_1 += 1  # Adjust for zero indexing

        self.neighbours = NearestNeighbors(n_neighbors=self.n_neighbours_1).fit(X)
        _, indices = self.neighbours.kneighbors(X)

        differences_X, differences_y = [], []
        for i, indice in enumerate(indices):
            for neighbor_index in indice[1:]:  # Skip the first one (itself)
                diff = np.abs(X[i] - X[neighbor_index])
                differences_X.append(diff)
                differences_y.append(y[i] - y[neighbor_index])

        self.train_X = X
        self.train_y = y
        self.classes_ = np.unique(y)

    def predict(self, X, model, dataset, input_shape):
        print("Before reshaping in fit, X shape:", X.shape) 
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
            sum_result = sum(
                self.train_y[i] + d for i, d in zip(indexes, differences)
                )
            avg_res = sum_result / self.n_neighbours_2
            y_pred.append(avg_res)

        return y_pred

    def get_params(self, deep=True):
        """
        Get parameters for the Linger Regressor.

        Parameters:
        - deep: If True, return parameters for the internal MLPRegressor.

        Returns:
        - params: Dictionary of parameters.

        """
        return {
            "n_neighbours_1": self.n_neighbours_1,
            "n_neighbours_2": self.n_neighbours_2,
        }

    def set_params(self, **params):
        """
        Set parameters for the Linger Regressor.

        Parameters:
        - params: Dictionary of parameters.

        Returns:
        - self: The modified Linger Regressor.

        """
        # Set parameters for your custom regressor
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
                # Check if the parameter also exists in the regressor
                if hasattr(self.regressor, param):
                    setattr(self.regressor, param, value)
            else:
                # If the parameter is not part of your custom regressor, set it in the internal MLPRegressor
                if hasattr(self.regressor, param):
                    setattr(self.regressor, param, value)
        return self
    
    def compute_pixelwise_difference(self, image1, image2):
        # Compute structural similarity index between the images
        ssim_index = structural_similarity(image1, image2, multichannel=True, win_size=3, data_range=image1.max() - image1.min())

        # Normalize SSIM index to [0, 1] range
        ssim_index_normalized = (ssim_index + 1) / 2

        # Compute pixel-wise difference as absolute difference between images
        difference = np.abs(image1 - image2)

        # Add normalized SSIM index as an additional feature
        difference = np.append(difference.flatten(), ssim_index_normalized)

        return difference