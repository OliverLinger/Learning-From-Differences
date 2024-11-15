from datetime import datetime
from sklearn.calibration import LabelEncoder
from sklearn.neural_network import MLPClassifier
from differences import _classification

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

LingerClassifier = _classification.LingerClassifier

df = pd.read_csv("datasets/raisin_data/Raisin_Dataset.csv")
from datetime import datetime
from sklearn.calibration import LabelEncoder
from sklearn.neural_network import MLPClassifier
from differences import _classification

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

LingerClassifier = _classification.LingerClassifier

LingerClassifier = _classification.LingerClassifier

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=2)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_data(df, features, numeric_features, nominal_features, columns):
    df.columns = columns
    df.dropna(subset=features, inplace=True)
    df.reset_index(drop=True, inplace=True)

    dev_df, test_df = train_test_split(df, train_size=0.8, random_state=2)
    copy_df = dev_df.copy()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler())]), numeric_features),
        ("nom", Pipeline([
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), nominal_features)],
        remainder="passthrough")

    dev_X = dev_df[features]
    test_X = test_df[features]

    dev_y = dev_df['Class'].values
    test_y = test_df['Class'].values

    label_encoder = LabelEncoder()
    dev_y = label_encoder.fit_transform(dev_y)
    test_y = label_encoder.transform(test_y)

    return dev_X, test_X, dev_y, test_y, preprocessor

def train_knn_classifier(dev_X, dev_y, preprocessor):
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", KNeighborsClassifier())
    ])

    knn_param_grid = {"predictor__n_neighbors": [2, 5, 7, 10, 13, 15, 17, 21]}

    knn_gs = GridSearchCV(knn_pipeline, knn_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=1)
    knn_gs.fit(dev_X, dev_y)

    return knn_gs

def train_knn_classifier_weighted(dev_X, dev_y, preprocessor):
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", KNeighborsClassifier(weights='distance'))
    ])

    knn_param_grid = {"predictor__n_neighbors": [2, 5, 7, 10, 13, 15, 17, 21]}

    knn_gs = GridSearchCV(knn_pipeline, knn_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=1)
    knn_gs.fit(dev_X, dev_y)

    return knn_gs

def train_neural_network(dev_X, dev_y, preprocessor):
    nn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", MLPClassifier())
    ])

    nn_param_grid = {
    "predictor__hidden_layer_sizes": [(256, 128), (128, 64), (200, 100), (300, 200, 100), (400, 300, 200, 100)],
    "predictor__activation": ["identity", "logistic", "tanh", "relu"],
    "predictor__alpha": [0.0001, 0.001, 0.01, 0.1],
    "predictor__max_iter": [1500, 2000],
    "predictor__early_stopping": [True],
    "predictor__validation_fraction": [0.1, 0.2, 0.3],
    "predictor__learning_rate_init": [0.001, 0.01, 0.1],
    "predictor__solver": ['adam', 'sgd'],
    "predictor__beta_1": [0.9, 0.95, 0.99],
    "predictor__beta_2": [0.999, 0.995, 0.9]
    }

    nn_gs = GridSearchCV(nn_pipeline, nn_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=8)
    nn_gs.fit(dev_X, dev_y)

    return nn_gs

def train_linger_classifier(dev_X, dev_y, preprocessor, best_nn_params):
    lfd_classifier_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", LingerClassifier())
    ])
    lfd_classifier_param_grid  = {}
    lfd_classifier_param_grid.update({
         "predictor__n_neighbours_1": [2, 5, 7, 10, 13, 15, 17, 21],
         "predictor__n_neighbours_2": [2, 5, 7, 10, 13, 15, 17, 21],
         "predictor__weighted_knn": [False],
         "predictor__additional_results_column": [True],
         "predictor__duplicated_on_distance": [False],
        "predictor__addition_of_context": [False],
    })
    # Update with best_nn_params
    lfd_classifier_param_grid.update(best_nn_params)
    for key, value in lfd_classifier_param_grid.items():
        if not isinstance(value, list):
            lfd_classifier_param_grid[key] = [value]

    lfd_classifier_gs = GridSearchCV(
        lfd_classifier_pipeline, lfd_classifier_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=8)

    lfd_classifier_gs.fit(dev_X, dev_y)

    return lfd_classifier_gs

def save_results(file_path, knn_classifier_gs, knn_classifier_gs_weighted, nn_gs, lfd_classifier_gs):
    with open(file_path, 'a') as file:
        file.write(f"Basic classifier, Addition of distance col Var3")
        file.write(f"Best Parameters KNN classifier: {knn_classifier_gs.best_params_,}\n")
        file.write(f"Best Score KNN classifier: {knn_classifier_gs.best_score_}\n")

        file.write(f"Best parameters for distance weighted KNN: {knn_classifier_gs_weighted.best_params_}")
        file.write(f"Best score for distance weighted KNN: {knn_classifier_gs_weighted.best_score_}")

        file.write(f"Best Parameters Neural net classifier: {nn_gs.best_params_,}\n")
        file.write(f"Best Score Neural net classifier: {nn_gs.best_score_}\n")

        file.write(f"Best Parameters Linger classifier: {lfd_classifier_gs.best_params_,}\n")
        file.write(f"Best Score Linger classifier: {lfd_classifier_gs.best_score_}\n")
        file.write("--------------------------------------------------------------\n")

def calculate_test_accuracies(file_path, knn_classifier_gs, knn_classifier_gs_weighted, nn_gs, lfd_classifier_gs, test_X, test_y):
    with open(file_path, 'a') as file:
        # Test the kNN classifier
        knn_test_accuracy = knn_classifier_gs.score(test_X, test_y)
        file.write(f"Test Accuracy for KNN classifier: {knn_test_accuracy}\n")

        # Test the Neural Network classifier
        nn_test_accuracy = nn_gs.score(test_X, test_y)
        file.write(f"Test Accuracy for Neural Network classifier: {nn_test_accuracy}\n")

        # Test the Linger Classifier
        lfd_classifier_test_accuracy = lfd_classifier_gs.score(test_X, test_y)
        file.write(f"Test Accuracy for Linger Classifier: {lfd_classifier_test_accuracy}\n")
        file.write("--------------------------------------------------------------\n")

def main():
    file_path = r'C:\Users\USER\final_year\fyp\results\RaisinResultsVar3.txt'
    df = pd.read_csv(r"C:\Users\USER\final_year\fyp\datasets\raisin_data\Raisin_Dataset_reduced.csv")
    columns = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
       'ConvexArea', 'Extent', 'Perimeter', 'Class']
    features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
       'ConvexArea', 'Extent', 'Perimeter']
    numeric_features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
       'ConvexArea', 'Extent', 'Perimeter']
    nominal_features = []
    dev_X, test_X, dev_y, test_y, preprocessor = preprocess_data(df, features, numeric_features, nominal_features, columns)

    knn_classifier_gs = train_knn_classifier(dev_X, dev_y, preprocessor)
    knn_classifier_gs_weighted = train_knn_classifier_weighted(dev_X, dev_y, preprocessor)
    nn_gs = train_neural_network(dev_X, dev_y, preprocessor)
    best_nn_params = nn_gs.best_params_
    lfd_classifier_gs = train_linger_classifier(dev_X, dev_y, preprocessor, best_nn_params)

    save_results(file_path, knn_classifier_gs, knn_classifier_gs_weighted, nn_gs, lfd_classifier_gs)
    calculate_test_accuracies(file_path, knn_classifier_gs, knn_classifier_gs_weighted, nn_gs, lfd_classifier_gs, test_X, test_y)
    print("Run complete")

if __name__ == "__main__":
    num_times_to_run = 2  # Change this to the desired number of iterations
    for _ in range(num_times_to_run):
        main()
