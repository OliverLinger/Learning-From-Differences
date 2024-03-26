# Authors: Oliver Linger

from collections import Counter
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier


class LingerClassifier(BaseEstimator, ClassifierMixin):
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
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples) if metric='precomputed'
                Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
                Target values.

        Returns
        -------
        self : LingerClassifier
        The fitted LearningFromDifferences classifier
        """
        self.n_neighbours_1 += 1  # Increment n_neighbors

        # Check if input matrices are sparse
        is_sparse_X = issparse(X)
        is_sparse_y = issparse(y)

        # Convert sparse matrices to dense arrays if needed
        if is_sparse_X:
            X = X.toarray()
        if is_sparse_y:
            y = y.toarray()

        neighbours = NearestNeighbors(n_neighbors=self.n_neighbours_1).fit(X)
        distances, indices = neighbours.kneighbors(X)

        differences_X = []
        differences_y = []

        """function for addition of base case context to the training data.
            The addition of the base case context."""
        # Addition of context
        if self.addition_of_context:
            print("Addition of context")
            differences_X, differences_y = self.addition_of_context_func_fit(
                X=X,
                y=y,
                indices=indices,
                differences_X=differences_X,
                differences_y=differences_y,
            )
        else:
            for i in indices:
                base = i[0]
                neighbors = i[1:]
                for n in neighbors:
                    differences_X.append(X[base] - X[n])
                    differences_y.append(y[base] - y[n])

        # If duplicate based on distance is activated
        if self.duplicated_on_distance:
            print("duplication occuring")
            differences_X, differences_y = self.duplicate_difference_based_on_distance(
                differences_X=differences_X,
                differences_y=differences_y,
                distances=distances,
            )

        # addition of distance column
        if self.additional_results_column:
            distances_X = []
            for i in distances:
                diffs = i[1:]
                for n in diffs:
                    distances_X.append(n)
            differences_X = self.add_additional_results_column_fit(
                differences_X=differences_X, distances=distances_X
            )

        self.classifier.fit(differences_X, differences_y)

        self.train_X = X
        self.train_y = y

        self.classes_ = np.unique(y)

    def predict(self, X):
        """
        Predict target values for the input data.

        Parameters:
        - X: Input data for prediction.

        Returns:
        - y_pred: Predicted target values.

        """
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbours_2).fit(self.train_X)
        distances, indices = nbrs.kneighbors(X)

        is_sparse_X = issparse(X)
        if is_sparse_X:
            X = X.toarray()

        differences_test_X = [
            X[test, :] - self.train_X[indices[test][nn_in_training], :]
            for test in range(len(X))
            for nn_in_training in range(len(indices[0]))
        ]

        """
        This section of code handles the retrieval of the distnaces for the nearest neighbors and appends them to the differences_test_X arrays.
        If self.additional_results_column is True:
            Since the classifier expects an additional column for the prediction, we iterate through the indices to perform the following steps:
        """
        if self.additional_results_column:
            print("Additional column implanted")
            distances_X = []
            for diffs in distances:
                for n in diffs:
                    distances_X.append(n)
            differences_test_X = self.add_additional_results_column_fit(
                differences_X=differences_test_X, distances=distances_X
            )

        """function for addition of base case context to the training data.
            The addition of the base case context."""
        # Addition of context
        if self.addition_of_context:
            differences_test_X = self.addition_of_context_func_pred(
                differences_test_X=differences_test_X, X=X
            )
            
        # makes a prediction based on the differences in the test set X
        predictions = self.classifier.predict(differences_test_X)

        # Finds the nearest neighbours of our predictions.
        predictions = [
            predictions[i : i + self.n_neighbours_2]
            for i in range(0, len(predictions), self.n_neighbours_2)
        ]
        y_pred = []
        if self.weighted_knn:
            for indexes, differences, distance in zip(
                indices, differences_test_X, distances
            ):
                weighted_results = {}

                # Calculate weighted results based on differences and distances
                for i, d, dist in zip(indexes, differences, distance):
                    result = self.train_y[i] + d
                    weight = 1 / (
                        dist + 1e-8
                    )  # Small constant added to avoid division by zero
                    if result in weighted_results:
                        weighted_results[result] += weight
                    else:
                        weighted_results[result] = weight

                # Retrieve the item with the highest weighted sum
                if weighted_results:
                    most_weighted_item = int(
                        max(weighted_results, key=lambda k: weighted_results[k])
                    )
                    y_pred.append(most_weighted_item)
                else:
                    # Handle the case where there are no items
                    y_pred.append(None)
            return y_pred
        # The non-weighted knn version.
        else:
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

    def add_additional_results_column_fit(self, differences_X, distances):
        """
        Adds an additional column with distance values associated with each differences_X during fitting.

        Parameters:
        - differences_X: List of feature differences.
        - distances: distances of the nearest neighours.

        Returns:
        - differences_X with an additional column.
        """
        for i, diff_res in enumerate(distances):
            diff_res = np.array([diff_res])
            differences_X[i] = np.concatenate((differences_X[i], diff_res))
        return differences_X

    def duplicate_difference_based_on_distance(
        self, differences_X, differences_y, distances
    ):
        """
        Duplicates differences based on the calculated distances.

        Parameters:
        - differences_X: List of feature differences.
        - differences_y: Target differences.
        - indices: Nearest neighbor indices.
        - distances: Distances to nearest neighbors.

        Returns:
        - Duplicated lists for differences_X and differences_y.
        """
        distances_X = []
        for d in distances:
            neighbors = d[1:]
            for n in neighbors:
                distances_X.append(n)
        # Convert the list to a numpy array for easier manipulation
        distances_array = np.array(distances_X)
        # Logarithmic transformation to compress the range of values
        log_values = np.log(distances_array + 1)  # Adding 1 to avoid logarithm of zero

        # Min-max scaling
        min_val = np.min(log_values)
        max_val = np.max(log_values)

        if min_val == max_val:
            # Handle case where min and max are the same
            scaled_distances_rounded = np.ones_like(log_values).astype(int)
        else:
            # Min-max scaling
            scaled_distances = 10 - ((log_values - min_val) / (max_val - min_val) * 10)
            # Round the scaled values to the nearest integer
            scaled_distances_rounded = np.round(scaled_distances).astype(int)

        # Filter out items with count 0 in dist
        filtered_data_X = [
            item
            for item, count in zip(differences_X, scaled_distances_rounded)
            if count != 0
        ]
        # Duplicate based on distance for differences_X
        duplicated_list_X = [
            item
            for item, count in zip(filtered_data_X, scaled_distances_rounded)
            for _ in range(count)
        ]

        filtered_data_y = [
            item
            for item, count in zip(differences_y, scaled_distances_rounded)
            if count != 0
        ]
        # Duplicate based on distance for differences_X
        duplicated_list_y = [
            item
            for item, count in zip(filtered_data_y, scaled_distances_rounded)
            for _ in range(count)
        ]
        # Retrun the duplicated lists
        return duplicated_list_X, duplicated_list_y

    def addition_of_context_func_fit(self, X, y, indices, differences_X, differences_y):
        for i in indices:
            base = i[0]
            neighbors = i[1:]
            for n in neighbors:
                base_case = X[base]
                diff = X[base] - X[n]
                combined_list = [item for pair in zip(diff, base_case) for item in pair]
                differences_X.append(np.array(combined_list))
                differences_y.append(y[base] - y[n])
        return differences_X, differences_y

    def addition_of_context_func_pred(self, differences_test_X, X):
        combined_differences_test_X = []
        duplicated_test_data = [item for item in X for _ in range(self.n_neighbours_2)]
        num_items = len(duplicated_test_data)
        for item_pos in range(num_items):
            combined_list = [
                item
                for pair in zip(
                    differences_test_X[item_pos], duplicated_test_data[item_pos]
                )
                for item in pair
            ]
            combined_differences_test_X.append(np.array(combined_list))
        return combined_differences_test_X