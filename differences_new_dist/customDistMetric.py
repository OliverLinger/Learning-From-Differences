import numpy as np
from sklearn.base import BaseEstimator
from collections import defaultdict

class CVDHMDistanceMetric(BaseEstimator):
    def __init__(self, numeric_features, categorical_features, features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.features = features
        self.vdm_values = defaultdict(lambda: defaultdict(dict))

    def fit(self, X, y):
        # Assuming X is a numpy array and y is the target array
        for i, feature_name in enumerate(self.features):
            if feature_name in self.categorical_features:
                unique_vals = np.unique(X[:, i])
                for val1 in unique_vals:
                    for val2 in unique_vals:
                        if val1 != val2:
                            self.vdm_values[i][val1][val2] = self.vdmf(y, X[:, i], val1, val2)
        return self

    def vdmf(self, y, feature_column, val1, val2):
        unique_classes = np.unique(y)
        numerator = 0
        denominator = 0

        for c in unique_classes:
            Nf_a_c = np.sum((feature_column == val1) & (y == c))
            Nf_b_c = np.sum((feature_column == val2) & (y == c))
            N_c = np.sum(y == c)

            if N_c == 0:
                continue

            prob_a_given_c = Nf_a_c / N_c
            prob_b_given_c = Nf_b_c / N_c

            numerator += abs(prob_a_given_c - prob_b_given_c)
            denominator += (prob_a_given_c + prob_b_given_c)

        if denominator == 0:
            return 0

        return numerator / denominator

    def single_sample_distance(self, x1, x2):
        total_dist_sq = 0.0
        for i, (feature1, feature2) in enumerate(zip(x1, x2)):
            if i in self.numeric_features:
                distance = min(1, abs(feature1 - feature2))
            elif i in self.categorical_features:
                distance = self.vdm_values[i].get(feature1, {}).get(feature2, 1)
            else:
                raise ValueError("Invalid feature index: {}".format(i))
            total_dist_sq += distance ** 2
        return np.sqrt(total_dist_sq)

    def __call__(self, x1, x2):
        return self.single_sample_distance(x1, x2)

# # Example usage:
# from sklearn.neighbors import NearestNeighbors

# # Assuming numeric_features, categorical_features, and features are properly defined:
# metric = CVDHMDistanceMetric(numeric_features=[0, 1], categorical_features=[2], features=[0, 1, 2])
# metric.fit(X_train, y_train)  # X_train and y_train need to be defined

# nn = NearestNeighbors(metric=metric)
# nn.fit(X_train)  # Use the custom metric for fitting
