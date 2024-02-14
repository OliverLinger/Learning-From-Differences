from datetime import datetime
from sklearn.neural_network import MLPRegressor
from differences import _regression

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

LingerRegressor = _regression.LingerRegressor


df = pd.read_csv("datasets/house_prices/dataset_corkB.csv")

# Define the file path
file_path = r'C:\Users\35383\4th_year\fyp\results\HousePricesResults.txt'

# Shuffle the dataset
df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)

features = ["flarea", "bdrms", "bthrms", "floors", "type", "devment", "ber", "location"]
numeric_features = ["flarea", "bdrms", "bthrms", "floors"]
nominal_features = ["type", "devment", "ber", "location"]

# Delete examples where flarea, devment or price are NaN
df.dropna(subset=["flarea", "bdrms", "devment", "price"], inplace=True)

# Delete examples whose floor areas are too small or too big
df = (df[(df["flarea"] >= 40) & (df["flarea"] < 750)]).copy()

# Delete examples whose prices are too high
df = (df[df["price"] < 2000]).copy()

# Reset the index
df.reset_index(drop=True, inplace=True)

# Split off the test set: 20% of the dataset.
dev_df, test_df = train_test_split(df, train_size=0.8, random_state=2)

# It can be good to do this on a copy of the dataset (excluding the test set, of course)
copy_df = dev_df.copy()

copy_df["room_size"] = copy_df["flarea"] / (copy_df["bdrms"] + copy_df["bthrms"])

# Create the preprocessor
preprocessor = ColumnTransformer([
        ("scaler", StandardScaler(), 
                numeric_features),
        ("nom", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")), 
                          ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), 
                nominal_features)],
        remainder="passthrough")

class InsertRoomSize(BaseEstimator, TransformerMixin):

    def __init__(self, insert=True):
        self.insert = insert
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.insert:
            X["room_size"] = X["flarea"] / (X["bdrms"] + X["bthrms"])
            
            # If the new feature is intended to replace the existing ones, 
            # you could drop the existing ones here
            # X.drop(["flarea", "bthrms", "bdrms"], axis=1)
    
            X = X.replace( [ np.inf, -np.inf ], np.nan )
        return X
    
preprocessor = ColumnTransformer([
        ("num", Pipeline([("room_size", InsertRoomSize()),
                          ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                          ("scaler", StandardScaler())]), 
                numeric_features),
        ("nom", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")), 
                          ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), 
                nominal_features)],
        remainder="passthrough")


# Extract the features but leave as a DataFrame
dev_X = dev_df[features]
test_X = test_df[features]

# Target values, converted to a 1D numpy array
dev_y = dev_df["price"].values
test_y = test_df["price"].values

# Create a pipeline that combines the preprocessor with kNN
knn = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsRegressor())])



# Create a dictionary of hyperparameters for kNN
knn_param_grid = {"predictor__n_neighbors": [2, 5, 7, 10, 13, 15, 17, 21],
                  "preprocessor__num__room_size__insert": [True]}

# Create the grid search object which will find the best hyperparameter values based on validation error
knn_gs = GridSearchCV(knn, knn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True, n_jobs=1)

# Run grid search by calling fit. It will also re-train on train+validation using the best parameters.
knn_gs.fit(dev_X, dev_y)

# Let's see how well we did
print(knn_gs.best_params_, knn_gs.best_score_)

KNN_weighted = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsRegressor(weights='distance'))])

# Create a dictionary of hyperparameters for kNN
knn_param_grid = {"predictor__n_neighbors": [2, 5, 7, 10, 13, 15, 17, 21]}

# Create the grid search object which will find the best hyperparameter values based on validation error
knn_weighted = GridSearchCV(KNN_weighted, knn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True, n_jobs=1)

# Run grid search by calling fit. It will also re-train on train+validation using the best parameters.
knn_weighted.fit(dev_X, dev_y)

# Let's see how well we did
print(f"Best parameters for distance weighted KNN: {knn_weighted.best_params_}")
print(f"Best score for distance weighted KNN: {knn_weighted.best_score_}")






# Create a pipeline with the preprocessor and neural network
nn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", MLPRegressor())
])

# Create a dictionary of hyperparameters for the neural network
nn_param_grid = {
    "predictor__hidden_layer_sizes": [(256, 128), (128, 64), (100,)],
    "predictor__activation": ['relu', 'tanh', 'logistic'],
    "predictor__solver": ['adam', 'sgd'],
    "predictor__alpha": [0.0001, 0.001, 0.01],
    "predictor__batch_size": ['auto', 32, 64],
    "predictor__learning_rate": ['constant', 'adaptive'],
    "predictor__learning_rate_init": [0.0001, 0.001, 0.01],
    "predictor__power_t": [0.3, 0.5, 0.7],
    "predictor__max_iter": [100, 200, 250, 300, 500, 1000, 1200],
    "predictor__shuffle": [True, False],
    "predictor__random_state": [None, 42],
    "predictor__tol": [1e-4, 1e-3, 1e-2],
    "predictor__verbose": [True, False],
    "predictor__warm_start": [True, False],
    "predictor__momentum": [0.9, 0.95, 0.99],
    "predictor__nesterovs_momentum": [True, False],
    "predictor__early_stopping": [True, False],
    "predictor__validation_fraction": [0.1, 0.2, 0.3],
    "predictor__beta_1": [0.8, 0.9, 0.95],
    "predictor__beta_2": [0.99, 0.999],
    "predictor__epsilon": [1e-8, 1e-7, 1e-6],
    "predictor__n_iter_no_change": [5, 10, 15],
    "predictor__max_fun": [10000, 15000, 20000],
    "preprocessor__num__room_size__insert": [True, False]
}

# Create the grid search object for the neural network
nn_gs = GridSearchCV(nn_pipeline, nn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True, n_jobs=1)
nn_gs.fit(dev_X, dev_y)

# Print the best parameters and score for the neural network
print("Best Parameters for Neural Network:", nn_gs.best_params_)
print("Best Score for Neural Network:", nn_gs.best_score_)










# Create a pipeline with the preprocessor and LingerRegressor
lfd_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LingerRegressor())
])

# Create a dictionary of hyperparameters for LingerRegressor
lfd_param_grid = {
    "predictor__hidden_layer_sizes": [(256, 128), (128, 64), (100,)],
    "predictor__n_neighbours_1": [2, 5, 7, 10, 13, 15, 21],
    "predictor__n_neighbours_2": [2, 5, 7, 10, 13, 15, 21],
    "predictor__max_iter": [100, 200, 250, 300, 500, 1000, 1200],
    "predictor__weighted_knn": [True, False],
    "predictor__additional_results_column": [True, False],
    "predictor__duplicated_on_distance": [True, False],
    "predictor__addition_of_context": [True, False],
    "predictor__alpha": [0.0001, 0.001, 0.01],
    "predictor__batch_size": ["auto", 32, 64],
    "predictor__learning_rate": ["constant", "adaptive"],
    "predictor__learning_rate_init": [0.0001, 0.001, 0.01],
    "predictor__power_t": [0.3, 0.5, 0.7],
    "predictor__shuffle": [True, False],
    "predictor__verbose": [True, False],
    "predictor__warm_start": [True, False],
    "predictor__momentum": [0.9, 0.95, 0.99],
    "predictor__nesterovs_momentum": [True, False],
    "predictor__early_stopping": [True, False],
    "predictor__validation_fraction": [0.1, 0.2, 0.3],
    "predictor__beta_1": [0.8, 0.9, 0.95],
    "predictor__beta_2": [0.99, 0.999],
    "predictor__epsilon": [1e-8, 1e-7, 1e-6],
    "predictor__n_iter_no_change": [5, 10, 15],
    "predictor__max_fun": [10000, 15000, 20000],
    "predictor__duplicated_on_distance": [True, False],
}

# Create the grid search object
lfd_gs = GridSearchCV(lfd_pipeline, lfd_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True, n_jobs=1)

# Run grid search by calling fit. It will also re-train on train+validation using the best parameters.

lfd_gs.fit(dev_X, dev_y)

# Let's see how well we did
print(lfd_gs.best_params_, lfd_gs.best_score_)

# Open the file in write mode and write the content
with open(file_path, 'a') as file:
    file.write(f"Test Time: {datetime.now().time()}\n")
    file.write(f"Best Parameters KNN regression: {knn_gs.best_params_,}\n")
    file.write(f"Best Score KNN regression: {knn_gs.best_score_}\n")

    file.write(f"Best parameters for distance weighted KNN: {knn_weighted.best_params_}")
    file.write(f"Best score for distance weighted KNN: {knn_weighted.best_score_}")

    file.write(f"Best Parameters Neural net regression: {nn_gs.best_params_,}\n")
    file.write(f"Best Score Neural net regression: {nn_gs.best_score_}\n")

    file.write(f"Best Parameters Linger regression: {lfd_gs.best_params_,}\n")
    file.write(f"Best Score Linger Regression: {lfd_gs.best_score_}\n")
    file.write("--------------------------------------------------------------\n")

# Test the kNN regressor
knn_test_accuracy = knn_gs.score(test_X, test_y)
print(f"Test Accuracy for KNN regressor: {knn_test_accuracy}")

# Test the Neural Network regressor
nn_test_accuracy = nn_gs.score(test_X, test_y)
print(f"Test Accuracy for Neural Network regressor: {nn_test_accuracy}")

# Test the Linger regressor
lfd_classifier_test_accuracy = lfd_gs.score(test_X, test_y)
print(f"Test Accuracy for Linger Regressor: {lfd_classifier_test_accuracy}")

with open(file_path, 'a') as file:
    # Test the kNN regressor
    knn_test_accuracy = knn_gs.score(test_X, test_y)
    file.write(f"Test Accuracy for KNN regressor: {knn_test_accuracy}\n")

    # Test the weighted kNN regressor
    knn_weighted_test_accuracy = knn_weighted.score(test_X, test_y)
    file.write(f"Test Accuracy for Weighted KNN regressor: {knn_weighted_test_accuracy}\n")

    # Test the Neural Network regressor
    nn_test_accuracy = nn_gs.score(test_X, test_y)
    file.write(f"Test Accuracy for Neural Network regressor: {nn_test_accuracy}\n")

    # Test the Linger Regressor
    lfd_classifier_test_accuracy = lfd_gs.score(test_X, test_y)
    file.write(f"Test Accuracy for Linger Regressor: {lfd_classifier_test_accuracy}\n")
    file.write("--------------------------------------------------------------\n")

print(f"Results have been saved to {file_path}")