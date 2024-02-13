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

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
from datetime import datetime
LingerRegressor = _regression.LingerRegressor

df = pd.read_csv("datasets/abalone/abalone.csv")

# Define the file path
file_path = r'C:\Users\35383\4th_year\fyp\results\AbaloneResults.txt'

# Shuffle the dataset
df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)


features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                        'Shucked weight', 'Viscera weight', 'Shell weight']
numeric_features = ['Length', 'Diameter', 'Height', 'Whole weight',
                    'Shucked weight', 'Viscera weight', 'Shell weight']
nominal_features = ['Sex']

# The columns aren't set so I assign the appropriate columns.
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

df.dropna(subset=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'], inplace=True)

# Reset the index
df.reset_index(drop=True, inplace=True)

# Split off the test set: 20% of the dataset.
dev_df, test_df = train_test_split(df, train_size=0.8, random_state=2)

# It can be good to do this on a copy of the dataset (excluding the test set, of course)
copy_df = dev_df.copy()


# Create the preprocessor
preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
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
dev_y = dev_df['Rings'].values
test_y = test_df['Rings'].values

# Create a pipeline that combines the preprocessor with kNN
knn = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsRegressor())])


# Create a dictionary of hyperparameters for kNN
knn_param_grid = {"predictor__n_neighbors": [19]}

# Create the grid search object which will find the best hyperparameter values based on validation error
knn_gs = GridSearchCV(knn, knn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True)

# Run grid search by calling fit. It will also re-train on train+validation using the best parameters.
knn_gs.fit(dev_X, dev_y)

# Let's see how well we did
print(knn_gs.best_params_, knn_gs.best_score_)

KNN_weighted = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsRegressor(weights='distance'))])

# Create a dictionary of hyperparameters for kNN
knn_param_grid = {"predictor__n_neighbors": [19]}

# Create the grid search object which will find the best hyperparameter values based on validation error
knn_weighted = GridSearchCV(KNN_weighted, knn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True)

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
# nn_param_grid = {
#     "predictor__hidden_layer_sizes": [(256, 128)],
#     "predictor__activation": ['relu']
# }

# # Create the grid search object for the neural network
# nn_gs = GridSearchCV(nn_pipeline, nn_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True)
# nn_gs.fit(dev_X, dev_y)

# # Print the best parameters and score for the neural network
# print("Best Parameters for Neural Network:", nn_gs.best_params_)
# print("Best Score for Neural Network:", nn_gs.best_score_)





# Create a pipeline with the preprocessor and LingerRegressor
lfd_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LingerRegressor())
])

# Create a dictionary of hyperparameters for LingerRegressor
lfd_param_grid = {
    "predictor__hidden_layer_sizes": [(256, 128)],
    "predictor__n_neighbours_1": [2],
    "predictor__n_neighbours_2": [2],
    "predictor__max_iter": [250],
    "predictor__weighted_knn": [True],
    "predictor__additional_results_column": [False],
    "predictor__duplicated_on_distance": [False],
    "predictor__addition_of_context": [True],
    # Include other hyperparameters for LingerRegressor here
}

# Create the grid search object
lfd_gs = GridSearchCV(lfd_pipeline, lfd_param_grid, scoring="neg_mean_absolute_error", cv=10, refit=True)

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

    # file.write(f"Best Parameters Neural net regression: {nn_gs.best_params_,}\n")
    # file.write(f"Best Score Neural net regression: {nn_gs.best_score_}\n")

    file.write(f"Best Parameters Linger regression: {lfd_gs.best_params_,}\n")
    file.write(f"Best Score Linger Regression: {lfd_gs.best_score_}\n")
    file.write("--------------------------------------------------------------\n")

# print(f"Results have been saved to {file_path}")

# # Test the kNN regressor
# knn_test_accuracy = knn_gs.score(test_X, test_y)
# print(f"Test Accuracy for KNN regressor: {knn_test_accuracy}")

# # Test the Neural Network regressor
# nn_test_accuracy = nn_gs.score(test_X, test_y)
# print(f"Test Accuracy for Neural Network regressor: {nn_test_accuracy}")

# # Test the Linger regressor
# lfd_classifier_test_accuracy = lfd_gs.score(test_X, test_y)
# print(f"Test Accuracy for Linger Regressor: {lfd_classifier_test_accuracy}")

# with open(file_path, 'a') as file:
#     # Test the kNN regressor
#     knn_test_accuracy = knn_gs.score(test_X, test_y)
#     file.write(f"Test Accuracy for KNN regressor: {knn_test_accuracy}\n")

#     # Test the Neural Network regressor
#     nn_test_accuracy = nn_gs.score(test_X, test_y)
#     file.write(f"Test Accuracy for Neural Network regressor: {nn_test_accuracy}\n")

#     # Test the Linger Regressor
#     lfd_classifier_test_accuracy = lfd_gs.score(test_X, test_y)
#     file.write(f"Test Accuracy for Linger Regressor: {lfd_classifier_test_accuracy}\n")
#     file.write("--------------------------------------------------------------\n")