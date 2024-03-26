import numpy as np


class CustomDistanceMetric:
    def __init__(self, cvdhm_distance_metric):
        self.cvdhm_distance_metric = cvdhm_distance_metric

    def __call__(self, X, Y):
        distances = []
        for x in X:
            row_distances = []
            for y in Y:
                dist = self.cvdhm_distance_metric.case_diff(x, y)
                row_distances.append(dist)
            distances.append(row_distances)
        return np.array(distances)