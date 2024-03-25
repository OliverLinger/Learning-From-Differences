from typing import List
from collections import defaultdict


class CVDHMDistanceMetric:
    def __init__(self, numeric_features: List, categorical_features: List, features: List):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.features = features
        self.vdm_values = defaultdict(dict)

    def case_diff(self, C1, C2, y, feature_values):
        total_dist_sq = 0.0
        for feature1, feature2, f in zip(C1, C2, self.features):
            dist_sq = self.CaseVDHMi(feature1, feature2, self.numeric_features, self.categorical_features, f, y, feature_values) ** 2
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
        for c in set(y):
            Nf_a_c = sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val1)
            Nf_b_c = sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val2)
            numerator += abs(Nf_a_c / sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val1) -
                              Nf_b_c / sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val2))
            denominator += abs(Nf_a_c / sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val1) -
                                Nf_b_c / sum(1 for i in range(len(y)) if y[i] == c and feature_values[i] == val2))
        if denominator == 0:
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
