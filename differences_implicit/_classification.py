# Authors: Oliver Linger

from collections import Counter, defaultdict
import random
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier


class LingerImplicitClassifier(BaseEstimator, ClassifierMixin):
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
        random_pairs=False,
        single_pair=False,
    ):
        """
        Linger EnsembleClassifier is a custom classifier based on k-nearest neighbors and N specialised MLPClassifier.
        Where N is the number of classes in the dataset.

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
        self.classes_ = []
        self.random_pairs = random_pairs
        self.single_pair = single_pair

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
        # Check if input matrices are sparse
        is_sparse_X = issparse(X)
        is_sparse_y = issparse(y)

        # Convert sparse matrices to dense arrays if needed
        if is_sparse_X:
            X = X.toarray()
        if is_sparse_y:
            y = y.toarray()
            
        self.classes_ = np.unique(y)
        self.adaptation_networks_ = {}

        # Get nearest neighbour for each case in X and y.
        neighbours = NearestNeighbors(n_neighbors=2).fit(X)
        distances, indices = neighbours.kneighbors(X)
        case_pairs_X = []
        case_pairs_y = []
        for i in indices:
                base = i[0]
                neighbors = i[1:]
                for n in neighbors:
                    case_x = [X[base],  X[n]]
                    case_y = [y[base], y[n]]
                    case_pairs_X.append(case_x)
                    case_pairs_y.append(case_y)
        # problem is the base case, solution is the retrieved case.
        
        
        # Grouping case pairs based on source solutions
        self.case_pairs_by_class = defaultdict(list)
        self.cases_without_concatenation = defaultdict(list)
        self.case_pairs_y = defaultdict(list)
        for sources, targets in zip(case_pairs_X, case_pairs_y):
            self.cases_without_concatenation[targets[1]].append(sources[0])
            # The retrived case is the solution 
            problem_query = np.array(sources[0])
            retrieved_problem = np.array(sources[1])
            case_pair_problem = np.concatenate((problem_query, retrieved_problem))
            self.case_pairs_by_class[targets[1]].append(case_pair_problem)
            self.case_pairs_y[targets[1]].append(targets[0])
        # Training adaptation neural networks for each clas
        for class_label, cases in self.case_pairs_by_class.items():
            self.adaptation_networks_[class_label] = MLPClassifier(
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
            self.adaptation_networks_[class_label].fit(cases, self.case_pairs_y[class_label])
    
    def predict(self, X):
        predictions = []
        is_sparse_X = issparse(X)

        # Convert sparse matrices to dense arrays if needed
        if is_sparse_X:
            X = X.toarray()

        # Create nearest neighbor finders for each class
        nbrs_by_class = {}
        for class_label in self.classes_:
            if self.single_pair:
                nbrs = NearestNeighbors(n_neighbors=1).fit(self.cases_without_concatenation[class_label])
            else:
                nbrs = NearestNeighbors(n_neighbors=self.n_neighbours_2).fit(self.cases_without_concatenation[class_label])
            nbrs_by_class[class_label] = nbrs

        # Find nearest neighbors and predict for each test sample
        for i, sample in enumerate(X):
            # Collect predictions from each class's nearest neighbors
            class_predictions = {class_label: [] for class_label in self.classes_}

            # Find nearest neighbors for each class
            for class_label in self.classes_:
                if self.random_pairs:
                    indices = random.sample(range(len(self.cases_without_concatenation[class_label])), self.n_neighbours_2)
                else:
                    distances, indices = nbrs_by_class[class_label].kneighbors([sample])
                nearest_neighbors = indices[0]  # Nearest neighbors indices for the current sample

                # Adapt nearest neighbor cases and collect predictions
                for neighbor_index in nearest_neighbors:
                    neighbor_sample = self.cases_without_concatenation[class_label][neighbor_index]
                    # Adapt the nearest neighbor case using the corresponding adaptation network
                    query = np.array(sample)
                    retrived = np.array(neighbor_sample)
                    merged_case_pair = [np.concatenate((retrived, query))]
                    adapted_prediction = self.adaptation_networks_[class_label].predict(merged_case_pair)
                    class_predictions[class_label].append(adapted_prediction[0])
                    
            # Perform majority voting to decide the final classification
            all_predictions = [prediction for predictions in class_predictions.values() for prediction in predictions]
            prediction_counter = Counter(all_predictions)
            most_common_predictions = [prediction for prediction, count in prediction_counter.items() if count == prediction_counter.most_common(1)[0][1]]
            # If there is a tie-break, randomly choose one prediction, otherwise choose the most common one
            if len(most_common_predictions) > 1:
                most_common_prediction = random.choice(most_common_predictions)
            else:
                most_common_prediction = most_common_predictions[0]
            predictions.append(most_common_prediction)
        return predictions
    
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