from collections import Counter
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

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
        Initialize the DifferenceImageGenerator.

        Parameters:
        - n_neighbours: Number of nearest neighbors to consider for computing difference images.
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

    def predict(self, X, model, dataset):
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
        if dataset == "cifar":
            differences_test_X = diff_X.reshape(-1, 32, 32, 3)  # Assuming RGB 
        if dataset == "mnist":
            differences_test_X = diff_X.reshape(-1, 28, 28, 1)  # Assuming RGB 
        # Make a prediction based on the differences in the test set X
        predictions = model.predict(differences_test_X)
        
        y_pred = []
        for indexes, differences in zip(indices, predictions):
            results = [self.train_y[i][0] + d for i, d in zip(indexes, differences)]
            print(results[:10])
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
