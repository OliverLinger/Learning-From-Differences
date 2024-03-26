# Authors: Oliver Linger

from collections import Counter
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from skimage.metrics import structural_similarity


class LingerImageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_neighbours_1=5,
        n_neighbours_2=5,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        weighted_knn=False,
        additional_results_column=False,
        duplicated_on_distance=False,
        addition_of_context=False,
    ):
        """
        Linger Classifier is a custom classifier based on k-nearest neighbors and MLPClassifier.

        Parameters:
        - n_neighbours_1: Number of neighbors for the first k-nearest neighbors search during fitting.
        - n_neighbours_2: Number of neighbors for the second k-nearest neighbors search during prediction.
        - hidden_layer_sizes: Tuple defining the size of hidden layers in the MLPClassifier.
        - activation: Activation function for MLPClassifier.
        - solver: Solver for MLPClassifier.
        - ... (other parameters for MLPClassifier)

        Attributes:
        - classifier: Internal MLPClassifier used for training and prediction.
        - train_X, train_y: Training data stored after fitting.

        """
        self.n_neighbours_1 = n_neighbours_1
        self.n_neighbours_2 = n_neighbours_2
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.train_X = []
        self.train_y = []
        self.weighted_knn = weighted_knn
        self.additional_results_column = additional_results_column
        self.duplicated_on_distance = duplicated_on_distance
        self.addition_of_context = addition_of_context

        self.classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change,
            max_fun=self.max_fun,
        )
    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
        
        # Increment n_neighbors
        self.n_neighbours_1 += 1  

        # Flatten images into vectors
        X_flat = np.reshape(X, (X.shape[0], -1))

        # Fit k-nearest neighbors
        neighbours = NearestNeighbors(n_neighbors=self.n_neighbours_1).fit(X_flat)
        distances, indices = neighbours.kneighbors(X_flat)

        differences_X = []
        differences_y = []

        for i in indices:
            base = i[0]
            neighbors = i[1:]
            for n in neighbors:
                # Compute pixel-wise differences between images
                diff = self.compute_pixelwise_difference(X[base], X[n])
                differences_X.append(diff)
                differences_y.append(np.abs(y[base] - y[n]))
        self.classifier.fit(np.array(differences_X), np.array(differences_y))

        self.train_X = X
        self.train_y = y

        self.classes_ = np.unique(y)

    def predict(self, X):
        # Fit k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbours_2).fit(self.train_X)
        distances, indices = nbrs.kneighbors(np.reshape(X, (X.shape[0], -1)))

        differences_test_X = [
            self.compute_pixelwise_difference(X[test], self.train_X[indices[test][i]])
            for test in range(len(X))
            for i in range(len(indices[0]))
        ]
        # Predict using the model
        predictions = self.classifier.predict(np.array(differences_test_X))
        # Finds the nearest neighbours of our predictions.
        predictions = [
            predictions[i : i + self.n_neighbours_2]
            for i in range(0, len(predictions), self.n_neighbours_2)
        ]
        y_pred = []
        for indexes, differences in zip(indices, predictions):
            results = [self.train_y[i] + d for i, d in zip(indexes, differences)]
            counts = Counter(results)
            # Retrieve the most common item
            most_common_item = counts.most_common(1)
            if most_common_item:
                y_pred.append(most_common_item[0][0])
            else:
                # Handle the case where there are no items
                y_pred.append(None)
        return y_pred

    
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
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "learning_rate_init": self.learning_rate_init,
            "power_t": self.power_t,
            "max_iter": self.max_iter,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "tol": self.tol,
            "verbose": self.verbose,
            "warm_start": self.warm_start,
            "momentum": self.momentum,
            "nesterovs_momentum": self.nesterovs_momentum,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "n_iter_no_change": self.n_iter_no_change,
            "max_fun": self.max_fun,
            "weighted_knn": self.weighted_knn,
            "additional_results_column": self.additional_results_column,
            "duplicated_on_distance": self.duplicated_on_distance,
            "addition_of_context": self.addition_of_context,
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
