from pickle import dump
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Had to supress 
import warnings
warnings.filterwarnings("ignore")

class LogisticalRegressionModel:
    def __init__(self, df: pd.DataFrame, training_features: List, 
    target_feature: str, numerical_features: List, nominal_features: List) -> None:
        self.numerical_features = numerical_features
        self.nominal_features = nominal_features
        self.df = df
        self.dev_df = pd.DataFrame
        self.test_df = pd.DataFrame
        self.y = pd.Series()
        self.training_features = training_features 
        self.target_feature = target_feature
        self.dev_y = pd.DataFrame
        self.test_y = pd.DataFrame

    def shuffle_df(self) -> None: 
        idx = np.random.permutation(self.df.index)
        self.df.reindex(idx)
        self.y.reindex(idx)
        self.df.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

    def seperate_training_target_data(self)-> None:
        self.df.columns = self.training_features
        label_encoder = preprocessing.LabelEncoder()
        self.df[self.target_feature] = label_encoder.fit_transform(
            self.df[self.target_feature])
        self.y = self.df[self.target_feature]
        self.df = self.df.drop(columns=self.target_feature)
        

    def describe_dataset(self) -> None:
        self.df.info()
        print(self.df.describe(include="all"))
    
    def column_or_1d(self, y, warn):
        y = np.ravel(y)
        return y

    def split_dataset(self) -> None:
        self.y = self.column_or_1d(self.y, warn=True)
        self.dev_df, self.test_df, self.dev_y, self.test_y = train_test_split(self.df, 
                                                                              self.y, 
                                                          train_size=0.8, 
                                                          stratify=self.y, 
                                                          random_state=4)

        
    def preprocessor(self):  
        preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, 
                                                    strategy="mean")),
                          ("scaler", StandardScaler())]), 
                self.numerical_features),
        ("nom", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, 
                                                    strategy="most_frequent")), 
                          ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), 
                self.nominal_features)], remainder="passthrough")
        return preprocessor
    
    def multinominal_logistical_regr(self, preprocessor) -> tuple:
        # Create a pipeline that combines the preprocessor with multinomial logistic reg
        multinomial = Pipeline([
            ("preprocessor", preprocessor),
            ("predictor", LogisticRegression(penalty=None))])
        validation_error = np.mean(cross_val_score(multinomial, self.dev_df, self.dev_y, 
                                                   scoring="accuracy", cv=10))
        multinomial.fit(self.dev_df, self.dev_y)
        print(f"dev accuracy score on test set: "
              f"{accuracy_score(self.dev_y, multinomial.predict(self.dev_df))}")
        print(f"dev validation score: {validation_error}")
        return multinomial, validation_error

    def mlp_classifier(self):
        param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['adam'],
            'hidden_layer_sizes': [
             (15,),(16,),(17,),(18,)
             ], 
             'alpha': [1e-5],
             'max_iter': [300]
        }
       ]
        clf = GridSearchCV(MLPClassifier(), param_grid, cv=3,
                           scoring='accuracy')
        clf.fit(self.dev_df, self.dev_y)
        print(clf.best_params_)
        validation_error = np.mean(cross_val_score(clf, self.dev_df, self.dev_y, 
                                                   scoring="accuracy", cv=10))
        print(f"mlp validation error: {validation_error}")
        print(f"dev accuracy score on test set: "
              f"{accuracy_score(self.dev_y, clf.predict(self.dev_df))}")

    
    def deploy(self, model):
        print(f"deployed accuracy score:"
              f"{accuracy_score(self.y, model.predict(self.df))}")
    
    
class run_scales_classification:
    def __init__(self) -> None:
        self.nominal_features = []
        self.numeric_features = ['Left-Weight', 
                                  'Left-Distance', 
                                  'Right-Weight', 
                                  'Right-Distance']
        self.df = pd.read_csv("datasets/scales/balance-scale.csv")
        self.training_features = ['class name', 
                                  'Left-Weight', 
                                  'Left-Distance', 
                                  'Right-Weight', 
                                  'Right-Distance']
        self.target_feature = "class name"
        self.LRM = LogisticalRegressionModel(df=self.df, 
                                             training_features=self.training_features, 
                                             target_features=self.target_feature, 
                                             numerical_features=self.numeric_features, 
                                             nominal_features=self.nominal_features)
        


    def run(self):
        self.LRM.seperate_training_target_data()
        self.LRM.shuffle_df()
        self.LRM.describe_dataset()
        self.LRM.split_dataset()
        preprocessor = self.LRM.preprocessor()
        model, mlr_validation_error = self.LRM.multinominal_logistical_regr(
            preprocessor=preprocessor)
        self.LRM.deploy(model=model)
        self.LRM.mlp_classifier()

class RunWinesClssification:
    def __init__(self) -> None:
        self.nominal_features = []
        self.numeric_features = ["Alcohol", "Malicacid", 
                                  "Ash", "Alcalinity_of_ash", 
                                  "Magnesium", "Total_phenols", "Flavanoids", 
                                  "Nonflavanoid_phenols", "Proanthocyanins", 
                                  "Color_intensity", "Hue", 
                                  "OD280/OD315_of_diluted wines", "Proline"]
        self.df = pd.read_csv("datasets/wines/wine.csv")
        self.training_features = ["class", "Alcohol", "Malicacid", 
                                  "Ash", "Alcalinity_of_ash", 
                                  "Magnesium", "Total_phenols", "Flavanoids", 
                                  "Nonflavanoid_phenols", "Proanthocyanins", 
                                  "Color_intensity", "Hue", 
                                  "OD280/OD315_of_diluted wines", "Proline"]
        self.target_feature = "class"
        self.LRM = LogisticalRegressionModel(df=self.df, 
                                             training_features=self.training_features, 
                                             target_feature=self.target_feature, 
                                             numerical_features=self.numeric_features, 
                                             nominal_features=self.nominal_features)

class RunIrisClassification:
    def __init__(self) -> None:
        self.df = pd.read_csv("datasets/iris/iris.csv")
        self.numeric_features = ["sepal length", "sepal width", "petal length", 
                                 "petal width"]
        self.nominal_features = []
        self.training_features = ["sepal length", 
                                  "sepal width", "petal length", 
                                  "petal width", "class"]
        self.target_feature = "class"
        self.LRM = LogisticalRegressionModel(df=self.df, 
                                             training_features=self.training_features, 
                                             target_feature=self.target_feature, 
                                             numerical_features=self.numeric_features, 
                                             nominal_features=self.nominal_features)
        

    def run(self):
        self.LRM.seperate_training_target_data()
        self.LRM.shuffle_df()
        self.LRM.describe_dataset()
        self.LRM.split_dataset()
        preprocessor = self.LRM.preprocessor()
        model, mlr_validation_error = self.LRM.multinominal_logistical_regr(
            preprocessor=preprocessor)
        self.LRM.deploy(model=model)
        self.LRM.mlp_classifier()


RWC = RunWinesClssification()
RIC = RunIrisClassification()
RIC.run()
#RWC.run()
