from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from typing import List
from collections import defaultdict

class CVDHMDistanceMetric:
    def __init__(self, numeric_features: List, categorical_features: List, features: List):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.features = features
        self.vdm_values = defaultdict(dict)

    def case_diff(self, C1, C2, y):
        total_dist_sq = 0.0
        for feature1, feature2, f in zip(C1, C2, self.features):
            dist_sq = self.CaseVDHMi(feature1, feature2, self.numeric_features, self.categorical_features, f, y) ** 2
            total_dist_sq += dist_sq
        result = total_dist_sq / 2
        return result

    def CaseVDHMi(self, a, b, num_f, cat_f, f, Y):
        if a is None or b is None:
            return 1  # Return 1 if either a or b is unknown
        elif f in cat_f:
            return self.vdmf(Y, a, b)
        elif f in num_f:
            return min(1, abs(a - b))
        else:
            raise ValueError("Invalid feature type: {}".format(f))

    def vdmf(self, y, val1, val2):
        numerator = 0
        denominator = 0
        unique_classes = set(y)

        for c in unique_classes:
            Nf_a_c = sum(1 for i in range(len(y)) if y[i] == c and val1[i])
            Nf_b_c = sum(1 for i in range(len(y)) if y[i] == c and val2[i])
            N_c = sum(1 for i in range(len(y)) if y[i] == c)

            if N_c == 0:  # Handling division by zero
                continue

            prob_a_given_c = Nf_a_c / N_c
            prob_b_given_c = Nf_b_c / N_c

            numerator += abs(prob_a_given_c - prob_b_given_c)
            denominator += abs(prob_a_given_c) + abs(prob_b_given_c)

        if denominator == 0:  # Handling division by zero
            return 0

        return numerator / denominator

    def fit(self, X, y):
        for i, f in enumerate(self.features):
            if f in self.categorical_features:
                for val1 in set(X[:, i]):
                    self.vdm_values[i][val1] = {}
                    for val2 in set(X[:, i]):
                        if val1 != val2:
                            self.vdm_values[i][val1][val2] = self.vdmf(y, val1, val2)

    def predict(self, X, n_neighbors):
        distances = pairwise_distances(X, metric=self.distance)
        indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        return indices

    def distance(self, X1, X2):
        n_samples1, n_features = X1.shape
        n_samples2, _ = X2.shape

        distances = np.zeros((n_samples1, n_samples2))

        for i in range(n_samples1):
            for j in range(n_samples2):
                distances[i, j] = self.case_diff(X1[i], X2[j], self.categorical_features)

        return distances
