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

# Define the file path
file_path = r'C:\Users\35383\4th_year\fyp\results\RaisinResults.txt'

# Shuffle the dataset
df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)

features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
       'ConvexArea', 'Extent', 'Perimeter']

numeric_features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
       'ConvexArea', 'Extent', 'Perimeter']
nominal_features = []

# Delete examples where flarea, devment or price are NaN
df.dropna(subset=['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
       'ConvexArea', 'Extent', 'Perimeter'], inplace=True)

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
dev_y = dev_df['Class'].values
test_y = test_df['Class'].values

# Use LabelEncoder to encode the target variable
label_encoder = LabelEncoder()
dev_y = label_encoder.fit_transform(dev_y)
test_y = label_encoder.transform(test_y)


# Create a pipeline that combines the preprocessor with kNN
knn_classifier = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsClassifier())])

# Create a dictionary of hyperparameters for kNN classifier
knn_classifier_param_grid = {"predictor__n_neighbors": [7]}

# Create the grid search object which will find the best hyperparameter values based on a classification metric
knn_classifier_gs = GridSearchCV(
    knn_classifier, knn_classifier_param_grid, scoring="accuracy", cv=10, refit=True)

# Run grid search by calling fit. It will also re-train on train+validation using the best parameters.
knn_classifier_gs.fit(dev_X, dev_y)

# Let's see how well we did
print(knn_classifier_gs.best_params_, knn_classifier_gs.best_score_)

# Create a pipeline that combines the preprocessor with weighted kNN
knn_classifier_weighted = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsClassifier(weights='distance'))])

# Create a dictionary of hyperparameters for kNN classifier
knn_classifier_param_grid_weighted  = {"predictor__n_neighbors": [7]}

# Create the grid search object which will find the best hyperparameter values based on a classification metric
knn_classifier_gs_weighted = GridSearchCV(
    knn_classifier_weighted , knn_classifier_param_grid_weighted , scoring="accuracy", cv=10, refit=True)

# Run grid search by calling fit. It will also re-train on train+validation using the best parameters.
knn_classifier_gs_weighted.fit(dev_X, dev_y)

# Let's see how well we did
print(f"Best parameters for distance weighted KNN: {knn_classifier_gs_weighted.best_params_}")
print(f"Best score for distance weighted KNN: {knn_classifier_gs_weighted .best_score_}")




# Create a pipeline with the preprocessor and neural network
nn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", MLPClassifier())
])

# Create a dictionary of hyperparameters for the neural network
nn_param_grid = {
    "predictor__hidden_layer_sizes": [(64, 32)],
    "predictor__activation": ['relu'],
    "predictor__max_iter": [1200],
}

# Create the grid search object for the neural network
nn_gs = GridSearchCV(nn_pipeline, nn_param_grid, scoring="accuracy", cv=10, refit=True)
nn_gs.fit(dev_X, dev_y)

# Print the best parameters and score for the neural network
print("Best Parameters for Neural Network:", nn_gs.best_params_)
print("Best Score for Neural Network:", nn_gs.best_score_)




# Create a pipeline with the preprocessor and LingerClassifier
lfd_classifier_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LingerClassifier())
])

# Create a dictionary of hyperparameters for LingerClassifier
lfd_classifier_param_grid = {
    "predictor__n_neighbours_1": [4],
    "predictor__n_neighbours_2": [8, 9],
    "predictor__hidden_layer_sizes": [(64, 32)],
    "predictor__max_iter": [1000],
    "predictor__weighted_knn": [True, False],
    # Include other hyperparameters for LingerClassifier here
}

# Create the grid search object
lfd_classifier_gs = GridSearchCV(
    lfd_classifier_pipeline, lfd_classifier_param_grid, scoring="accuracy", cv=10, refit=True)

# Run grid search by calling fit. It will also re-train on train+validation using the best parameters.
lfd_classifier_gs.fit(dev_X, dev_y)

# Let's see how well we did
print("Best Parameters for Linger Model:", lfd_classifier_gs.best_params_)
print("Best Score for Linger Model:", lfd_classifier_gs.best_score_)


# Open the file in write mode and write the content
with open(file_path, 'a') as file:
    file.write(f"Test Time: {datetime.now().time()}\n")
    file.write(f"Best Parameters KNN classifier: {knn_classifier_gs.best_params_,}\n")
    file.write(f"Best Score KNN classifier: {knn_classifier_gs.best_score_}\n")

    file.writev(f"Best parameters for distance weighted KNN: {knn_classifier_gs_weighted.best_params_}")
    file.write(f"Best score for distance weighted KNN: {knn_classifier_gs_weighted .best_score_}")

    file.write(f"Best Parameters Neural net classifier: {nn_gs.best_params_,}\n")
    file.write(f"Best Score Neural net classifier: {nn_gs.best_score_}\n")

    file.write(f"Best Parameters Linger classifier: {lfd_classifier_gs.best_params_,}\n")
    file.write(f"Best Score Linger classifier: {lfd_classifier_gs.best_score_}\n")
    file.write("--------------------------------------------------------------\n")


    # Print a confirmation message
print(f"Results have been saved to {file_path}")

# Test the kNN classifier
knn_test_accuracy = knn_classifier_gs.score(test_X, test_y)
print(f"Test Accuracy for KNN classifier: {knn_test_accuracy}")

# Test the Neural Network classifier
nn_test_accuracy = nn_gs.score(test_X, test_y)
print(f"Test Accuracy for Neural Network classifier: {nn_test_accuracy}")

# Test the Linger Classifier
lfd_classifier_test_accuracy = lfd_classifier_gs.score(test_X, test_y)
print(f"Test Accuracy for Linger Classifier: {lfd_classifier_test_accuracy}")

with open(file_path, 'a') as file:
    # Test the kNN classifier
    file.write(f"Test Time: {datetime.now().time()}\n")
    knn_test_accuracy = knn_classifier_gs.score(test_X, test_y)
    file.write(f"Test Accuracy for KNN classifier: {knn_test_accuracy}\n")

    # Test the Neural Network classifier
    nn_test_accuracy = nn_gs.score(test_X, test_y)
    file.write(f"Test Accuracy for Neural Network classifier: {nn_test_accuracy}\n")

    # Test the Linger Classifier
    lfd_classifier_test_accuracy = lfd_classifier_gs.score(test_X, test_y)
    file.write(f"Test Accuracy for Linger Classifier: {lfd_classifier_test_accuracy}\n")
    file.write("--------------------------------------------------------------\n")