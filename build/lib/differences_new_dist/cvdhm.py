from typing import List
from collections import defaultdict


class CVDHMDistanceMetric:
    def __init__(self, numeric_features: List, categorical_features: List, features: List):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.features = features
        self.vdm_values = defaultdict(dict)

    def case_diff(self, C1, C2, y):
        print(C1)
        quit()
        total_dist_sq = 0.0
        for feature1, feature2, f in zip(C1, C2, self.features):
            dist_sq = self.CaseVDHMi(feature1, feature2, self.numeric_features, self.categorical_features, f, y) ** 2
            total_dist_sq += dist_sq
        result = total_dist_sq / 2
        return result

    def CaseVDHMi(self, a, b, num_f, cat_f, f, Y, feature_values):
        if a is None or b is None:
            return 1  # Return 1 if either a or b is unknown
        elif f in cat_f:
            if a is None or b is None:
                return 1  # Return 1 if f is categorical and a or b is not observed
            else:
                return self.vdmf(feature_values, Y, a, b)
        elif f in num_f:
            return min(1, abs(a - b))
        else:
            raise ValueError("Invalid feature type: {}".format(f))

    def vdmf(self, feature_values, y, val1, val2):
        numerator = 0
        denominator = 0
        unique_classes = set(y)

        for c in unique_classes:
            Nf_a_c = sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val1)
            Nf_b_c = sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val2)
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
                            self.vdm_values[i][val1][val2] = self.vdmf(X[:, i], y, val1, val2)

    def get_params(self):
        pass

    def set_params(self, **params):
        pass
