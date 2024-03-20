from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from differences_implicit import _classification
import pandas as pd
import numpy as np
from datetime import datetime

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
        ("num", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                          ("scaler", StandardScaler())]), numeric_features),
        ("nom", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                          ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), nominal_features)],
        remainder="passthrough")

    dev_X = dev_df[features]
    test_X = test_df[features]

    dev_y = dev_df['Class'].values
    test_y = test_df['Class'].values
    
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

def train_neural_network_classifier(dev_X, dev_y, preprocessor):
    nn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", MLPClassifier())
    ])

    nn_param_grid = {
        "predictor__hidden_layer_sizes": [(200,), (256, 128)],
        "predictor__activation": ["relu"],
        "predictor__alpha": [0.0001],
        "predictor__max_iter": [1000],
        # "predictor__early_stopping": [True],
        # "predictor__validation_fraction": [0.1],
    }

    nn_gs = GridSearchCV(nn_pipeline, nn_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=8)
    nn_gs.fit(dev_X, dev_y)

    return nn_gs

def train_linger_classifier(dev_X, dev_y, preprocessor, best_nn_params):
    LingerClassifier = _classification.LingerImplicitClassifier

    lfd_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("predictor", LingerClassifier())
    ])

    lfd_param_grid = {}
    lfd_param_grid.update({
        "predictor__random_pairs": [True, False],
        "predictor__single_pair": [True, False],
        "predictor__hidden_layer_sizes": [(256, 128)],
        "predictor__n_neighbours_1": [2],
        "predictor__n_neighbours_2": [5, 10],
        "predictor__max_iter": [1000],
    })
    # Update with best_nn_params
    lfd_param_grid.update(best_nn_params)
    for key, value in lfd_param_grid.items():
        if not isinstance(value, list):
            lfd_param_grid[key] = [value]

    lfd_gs = GridSearchCV(lfd_pipeline, lfd_param_grid, scoring="accuracy", cv=10, refit=True, n_jobs=8)
    lfd_gs.fit(dev_X, dev_y)

    return lfd_gs

def save_results(file_path, knn_gs, nn_gs, lfd_gs):
    with open(file_path, 'a') as file:
        file.write(f"Test Time: {datetime.now().time()}\n")
        file.write(f"Best Parameters KNN classifier: {knn_gs.best_params_,}\n")
        file.write(f"Best Score KNN classifier: {knn_gs.best_score_}\n")
        file.write(f"Best Parameters Linger classifier: {lfd_gs.best_params_,}\n")
        file.write(f"Best Score Linger Classifier: {lfd_gs.best_score_}\n")
        file.write(f"Best Parameters Basic Neural Network: {nn_gs.best_params_,}\n")
        file.write(f"Best Score Basic Neural Network: {nn_gs.best_score_}\n")
        file.write("--------------------------------------------------------------\n")

def calculate_test_accuracies(file_path, knn_gs, lfd_gs, nn_gs, test_X, test_y):
    knn_test_accuracy = knn_gs.score(test_X, test_y)
    nn_test_accuracy = nn_gs.score(test_X, test_y)
    lfd_classifier_test_accuracy = lfd_gs.score(test_X, test_y)

    with open(file_path, 'a') as file:
        file.write(f"Test Accuracy for KNN classifier: {knn_test_accuracy}\n")
        file.write(f"Test Accuracy for Linger Classifier: {lfd_classifier_test_accuracy}\n")
        file.write(f"Test Accuracy for Basic Neural Network: {nn_test_accuracy}\n")
        file.write("--------------------------------------------------------------\n")

    print(f"Results have been saved to {file_path}")

def main():
    file_path = r'C:\Users\35383\4th_year\fyp\results\GlassImplicitResults.txt'
    df = pd.read_csv("datasets/glass/glass.csv",index_col=0)
    columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
    features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    numeric_features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    nominal_features = []
    dev_X, test_X, dev_y, test_y, preprocessor = preprocess_data(df, features, numeric_features, nominal_features, columns)
    
    knn_gs = train_knn_classifier(dev_X, dev_y, preprocessor)
    nn_gs = train_neural_network_classifier(dev_X, dev_y, preprocessor)
    best_nn_params = nn_gs.best_params_
    lfd_gs = train_linger_classifier(dev_X, dev_y, preprocessor, best_nn_params)

    save_results(file_path, knn_gs, nn_gs, lfd_gs)
    calculate_test_accuracies(file_path, knn_gs, lfd_gs, nn_gs, test_X, test_y)

if __name__ == "__main__":
    main()
