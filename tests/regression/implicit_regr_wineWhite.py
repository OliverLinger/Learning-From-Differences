import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from differences_implicit import _regression, regression_to_class
from sklearn.neural_network import MLPClassifier, MLPRegressor
from datetime import datetime

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=2)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_data(df, features, numeric_features, nominal_features, columns):
    df.columns = columns
    df.dropna(subset=[
    "fixed acidity","volatile acidity", "citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",], inplace=True)
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

    dev_y = dev_df["quality"].values
    test_y = test_df["quality"].values
    
    return dev_X, test_X, dev_y, test_y, preprocessor

def train_knn_regressor(dev_X, dev_y, preprocessor):
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", KNeighborsRegressor())
    ])

    knn_param_grid = {"predictor__n_neighbors": [2, 5, 7, 10, 13, 15, 17, 21]}

    knn_gs = GridSearchCV(knn_pipeline, knn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True, n_jobs=1)
    knn_gs.fit(dev_X, dev_y)

    return knn_gs

def train_weighted_knn_regressor(dev_X, dev_y, preprocessor):
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", KNeighborsRegressor(weights='distance'))
    ])

    knn_param_grid = {"predictor__n_neighbors": [2, 5, 7, 10, 13, 15, 17, 21]}

    knn_gs = GridSearchCV(knn_pipeline, knn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True, n_jobs=1)
    knn_gs.fit(dev_X, dev_y)

    return knn_gs

def train_neural_network(dev_X, dev_y, preprocessor):
    regr_to_class = regression_to_class.RegressionToClassificationConverter(n_segments=3, equal_division=True)
    dev_y, unique_ranges, min_val, max_val = regr_to_class.transform(y=dev_y)
    nn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", MLPClassifier())
    ])

    nn_param_grid = {
    "predictor__hidden_layer_sizes": [(256, 128), (128, 64), (200, 100)],
    "predictor__activation": ["relu"],
    "predictor__alpha": [0.01],
    "predictor__max_iter": [2000],
    "predictor__early_stopping": [True],
    "predictor__validation_fraction": [0.1],
    "predictor__learning_rate_init": [0.01],
    "predictor__solver": ['sgd'],
    "predictor__beta_1": [0.9],
    "predictor__beta_2": [0.9]
    }

    nn_gs = GridSearchCV(nn_pipeline, nn_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=8)
    nn_gs.fit(dev_X, dev_y)

    return nn_gs

def train_linger_regressor(dev_X, dev_y, preprocessor, best_nn_params):
    regr_to_class = regression_to_class.RegressionToClassificationConverter(n_segments=3, equal_division=True)
    dev_y, unique_ranges, min_val, max_val = regr_to_class.transform(y=dev_y)
    LingerRegressor = _regression.LingerImplicitRegressor
    lfd_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", LingerRegressor())
    ])
    # Convert single values to lists
    lfd_param_grid = {}

    # Add other parameters to lfd_param_grid
    lfd_param_grid.update({
        "predictor__random_pairs": [True],
        "predictor__single_pair": [False],
        "predictor__n_neighbours_1": [2, 5, 7, 10, 13, 15, 17, 21],
        "predictor__n_neighbours_2": [2, 5, 7, 10, 13, 15, 17, 21],
    })
    # Update with best_nn_params
    lfd_param_grid.update(best_nn_params)
    for key, value in lfd_param_grid.items():
        if not isinstance(value, list):
            lfd_param_grid[key] = [value]
            
    lfd_gs = GridSearchCV(lfd_pipeline, lfd_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=8)
    lfd_gs.fit(dev_X, dev_y)

    return lfd_gs, unique_ranges, min_val, max_val

def save_results(file_path, knn_gs, weighted_knn_gs, nn_gs, lfd_gs):
    with open(file_path, 'a') as file:
        file.write(f"Best Parameters KNN regression: {knn_gs.best_params_,}\n")
        file.write(f"Best Score KNN regression: {knn_gs.best_score_}\n")
        file.write(f"Best Parameters weighted KNN regression: {weighted_knn_gs.best_params_,}\n")
        file.write(f"Best Score weighted KNN regression: {weighted_knn_gs.best_score_}\n")
        file.write(f"Best Parameters Linger regression: {lfd_gs.best_params_,}\n")
        file.write(f"Best Score Linger Regression: {lfd_gs.best_score_}\n")
        file.write(f"Best Parameters Basic Neural Network: {nn_gs.best_params_,}\n")
        file.write(f"Best Score Basic Neural Network: {nn_gs.best_score_}\n")
        file.write("--------------------------------------------------------------\n")

def calculate_test_accuracies(file_path, knn_gs,weighted_knn_gs, lfd_gs, nn_gs, test_X, test_y, unique_ranges, min_val, max_val):
    knn_test_accuracy = knn_gs.score(test_X, test_y)
    weighted_knn_test_accuracy = weighted_knn_gs.score(test_X, test_y)


    regr_to_class = regression_to_class.RegressionToClassificationConverter(n_segments=3, equal_division=True)
    test_y, unique_ranges, min_val, max_val = regr_to_class.transform(y=test_y, unique_ranges=unique_ranges, min_val=min_val, max_val=max_val)
    lfd_classifier_test_accuracy = lfd_gs.score(test_X, test_y)
    nn_test_accuracy = nn_gs.score(test_X, test_y)

    with open(file_path, 'a') as file:
        file.write(f"Test Accuracy for KNN regressor: {knn_test_accuracy}\n")
        file.write(f"Test Accuracy for weighted KNN regressor: {weighted_knn_test_accuracy}\n")
        file.write(f"Test Accuracy for Linger Regressor: {lfd_classifier_test_accuracy}\n")
        file.write(f"Test Accuracy for Basic Neural Network: {nn_test_accuracy}\n")
        file.write("--------------------------------------------------------------\n")

    print(f"Results have been saved to {file_path}")

def main():
    file_path = r'C:\Users\USER\final_year\fyp\results\regressionImplicit\WhiteWineResultsImplicitBasic.txt'
    df = pd.read_csv("datasets/wineQuality/winequality-white_Reduced.csv")
    columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
    features = [
    "fixed acidity","volatile acidity", "citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",]
    numeric_features = ["fixed acidity","volatile acidity", "citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
    nominal_features = []
    dev_X, test_X, dev_y, test_y, preprocessor = preprocess_data(df, features, numeric_features, nominal_features, columns)
    
    knn_gs = train_knn_regressor(dev_X, dev_y, preprocessor)
    weighted_knn_gs = train_weighted_knn_regressor(dev_X, dev_y, preprocessor)

    nn_gs = train_neural_network(dev_X, dev_y, preprocessor)
    best_nn_params = nn_gs.best_params_
    lfd_gs, unique_ranges,  min_val, max_val = train_linger_regressor(dev_X, dev_y, preprocessor, best_nn_params)

    save_results(file_path, knn_gs, weighted_knn_gs, nn_gs, lfd_gs)
    calculate_test_accuracies(file_path, knn_gs, weighted_knn_gs, lfd_gs, nn_gs, test_X, test_y, unique_ranges, min_val, max_val)

if __name__ == "__main__":
    num_times_to_run = 5  # Change this to the desired number of iterations
    for _ in range(num_times_to_run):
        main()