import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


class CustomDistanceMetric:
    def __init__(self, Y, feature_values):
        self.Y = Y
        self.feature_values = feature_values

    def custom_distance(self, x, y):
        # Calculate the distance using your custom distance metric
        return self.case_diff(x, y)  # Assuming case_diff is your desired custom distance function

    def case_diff(self, C1, C2):
        total_dist_sq = 0.0
        for feature1, feature2, f in zip(C1, C2, self.features):
            dist_sq = self.CaseVDHMi(feature1, feature2, f) ** 2
            total_dist_sq += dist_sq
        result = total_dist_sq / 2
        return result

    def CaseVDHMi(self, a, b, f):
        if a is None or b is None:
            return 1  # Return 1 if either a or b is unknown
        elif f in self.categorical_features:
            return self.vdmf(a, b)
        elif f in self.numeric_features:
            return min(1, abs(a - b))
        else:
            raise ValueError("Invalid feature type: {}".format(f))

    def vdmf(self, val1, val2):
        numerator = 0
        denominator = 0
        for c in set(self.Y):
            Nf_a_c = sum(1 for i in range(len(self.Y)) if self.Y[i] == c and self.feature_values[i] == val1)
            Nf_b_c = sum(1 for i in range(len(self.Y)) if self.Y[i] == c and self.feature_values[i] == val2)
            numerator += abs(Nf_a_c / sum(1 for i in range(len(self.Y)) if self.Y[i] == c and self.feature_values[i] == val1) -
                              Nf_b_c / sum(1 for i in range(len(self.Y)) if self.Y[i] == c and self.feature_values[i] == val2))
            denominator += abs(Nf_a_c / sum(1 for i in range(len(self.Y)) if self.Y[i] == c and self.feature_values[i] == val1) -
                                Nf_b_c / sum(1 for i in range(len(self.Y)) if self.Y[i] == c and self.feature_values[i] == val2))
        if denominator == 0:
            return 0
        return numerator / denominator


# Define your preprocess_data function
def preprocess_data(df, features, numeric_features, nominal_features, columns):
    df.columns = columns
    df.dropna(subset=features, inplace=True)
    df.reset_index(drop=True, inplace=True)

    dev_df, test_df = train_test_split(df, train_size=0.8, random_state=2)
    copy_df = dev_df.copy()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                          ("scaler", StandardScaler())]), numeric_features),
        ("nom", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                          ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), nominal_features)],
        remainder="passthrough")

    dev_X = dev_df[features]
    test_X = test_df[features]

    dev_y = dev_df["Rings"].values
    test_y = test_df["Rings"].values

    return dev_X, test_X, dev_y, test_y, preprocessor


# Define your train_nearest_neighbors function
def train_nearest_neighbors(dev_X, test_X, preprocessor, custom_distance_metric):
    # Fit and transform the training data
    dev_X_transformed = preprocessor.fit_transform(dev_X)

    # Initialize NearestNeighbors with custom distance metric
    nn = NearestNeighbors(n_neighbors=2, algorithm='brute', metric=custom_distance_metric.custom_distance)

    # Fit the model to the transformed training data
    nn.fit(dev_X_transformed)

    # Transform the test data
    test_X_transformed = preprocessor.transform(test_X)

    return nn, test_X_transformed


# Example usage
df = pd.read_csv("datasets/abalone/abalone.csv")
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
           'Rings']
features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
numeric_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
nominal_features = ['Sex']

dev_X, test_X, dev_y, test_y, preprocessor = preprocess_data(df, features, numeric_features, nominal_features, columns)

# Assuming you have defined Y and feature_values elsewhere
custom_distance_metric = CustomDistanceMetric(Y=dev_y, feature_values=dev_X)
nn_model, test_X_transformed = train_nearest_neighbors(dev_X, test_X, preprocessor, custom_distance_metric)

# Example usage to find nearest neighbors
distances, indices = nn_model.kneighbors([test_X_transformed[0]])
print("Nearest neighbors distances:", distances)
print("Nearest neighbors indices:", indices)
