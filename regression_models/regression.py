from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from typing import List
from joblib import dump
import pandas as pd
from pandas.plotting import scatter_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# I create a labelled data set so that the machine learning model will carry 
# out supervised learning
# Get the dataset as a dataframe.


class MetaTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer=None):
        self.transformer = transformer
        
    def fit(self, X, y=None):
        if self.transformer:
            self.transformer.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        if self.transformer:
            return self.transformer.transform(X)
        else:
            return X
        
class RegressionModel:
    def __init__(self, df: pd.DataFrame, nominal_features: List, numeric_features: List, 
                 training_features: List, target_feature: str) -> None:
        self.df = df
        self.dev_df = pd.DataFrame
        self.test_df = pd.DataFrame
        self.nominal_features = nominal_features 
        self.numeric_features = numeric_features 
        self.training_features = training_features 
        self.target_feature = target_feature
        self.y = pd.Series
        self.dev_y = pd.DataFrame
        self.test_y = pd.DataFrame
        self.dev_X = pd.DataFrame
        self.test_X = pd.DataFrame


    def shuffle_df(self) -> None: 
        # Shuffle the dataset
        self.df = self.df.sample(frac=1, random_state=2)
        self.df.reset_index(drop=True, inplace=True)

    
    def data_statistics(self) -> None:
        print(self.df.describe(include="all"))

    def column_or_1d(self, y, warn):
        y = np.ravel(y)
        return y
    
    def split_training_test_data(self) -> None:
        self.y = self.column_or_1d(self.y, warn=True)
        self.dev_df, self.test_df = train_test_split(self.df, train_size=0.8, random_state=2)


    def nominal_numerical_details(self) -> None:
        # This is for nominal features and there are no null values, No editing is needed
        for feature in self.nominal_features:
            print(feature, self.dev_df[feature].unique())
        for feature in self.numeric_features:
            print(feature, self.dev_df[feature].unique()) 

    def describe_data_visually(self, x_plane, y_plane) -> None:
        # This plot illustrates the correlation between the diameter of a snail and the
        #  number of rings it has.
        sns.scatterplot( x=x_plane, y=y_plane, data=self.dev_df)
        copy_df = self.dev_df.copy()
        sns.scatterplot(copy_df["Length"])
        sns.scatterplot(copy_df["Diameter"])
        plt.show()
 

    ''' dealing with data '''
    def remove_outliers(self)-> None:
        index_of_height_outlier = self.dev_df.Height[self.dev_df.Height>0.4].index
        self.dev_df.drop(index_of_height_outlier, inplace=True)


    def convert_nomminal_to_num(self) -> None:
        one_hot_enc_data = pd.get_dummies(self.dev_df, columns=self.nominal_features)
        self.dev_df = one_hot_enc_data
        return self.dev_df.columns


    def preprocessing(self, numeric_features: List, nominal_features: List):
        preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, 
                                                    strategy="mean")),
                          ("scaler", MetaTransformer())]), 
                numeric_features),
        ("nom", Pipeline([("imputer", SimpleImputer(missing_values=np.nan,
                                                     strategy="most_frequent")), 
                          ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), 
                nominal_features)], remainder="passthrough")
        return preprocessor
    

    def prepare_for_model_selection(self)-> None:
        # Extract the features but leave as a DataFrame
        self.dev_X = self.dev_df[self.training_features]
        self.test_X = self.test_df[self.training_features]
        
        # Target values, converted to a 1D numpy array
        self.dev_y = self.dev_df[self.target_feature].values
        self.test_y = self.test_df[self.target_feature].values
    

    def combine_preproc_with_knn(self, preprocessor, n_neighbours: List) -> tuple:
        # Create a pipeline that combines the preprocessor with kNN
        knn = Pipeline([
            ("preprocessor", preprocessor),
            ("predictor", KNeighborsRegressor())])
        
        # Create a dictionary of hyperparameters for kNN
        knn_param_grid = {"predictor__n_neighbors": n_neighbours,
                          "preprocessor__num__scaler__transformer": 
                          [StandardScaler(), MinMaxScaler(), RobustScaler()]}
        
        knn_gs = GridSearchCV(knn, knn_param_grid, scoring="neg_mean_absolute_error", 
                              cv=10, refit=True)

        knn_gs.fit(self.dev_X, self.dev_y)

        print(knn_gs.best_params_, knn_gs.best_score_)

        knn.set_params(**knn_gs.best_params_) 
        scores = cross_validate(knn, self.dev_X, self.dev_y, cv=10, 
                        scoring="neg_mean_absolute_error", return_train_score=True)
        print("Training error: ", np.mean(np.abs(scores["train_score"])))
        print("Validation error: ", np.mean(np.abs(scores["test_score"])))
        return knn_gs, knn



    def ridge_regression_pipeline(self, preprocessor, predictor_alpha: List):
        ridge = Pipeline([
            ("preprocessor", preprocessor),
            ("predictor", Ridge(solver="sag"))])
        ridge_param_grid = {"preprocessor__num__scaler__transformer": [StandardScaler(),
                                                                        MinMaxScaler(), RobustScaler()],
                     "predictor__alpha": predictor_alpha}
        ridge_gs = GridSearchCV(ridge, ridge_param_grid, scoring="neg_mean_absolute_error", 
                                cv=10, refit=True)
        ridge_gs.fit(self.dev_X, self.dev_y)

        # Let's see how well we did
        print(ridge_gs.best_params_, ridge_gs.best_score_)
        ridge.set_params(**ridge_gs.best_params_) 
        scores = cross_validate(ridge, self.dev_X, self.dev_y, cv=10, 
                        scoring="neg_mean_absolute_error", return_train_score=True)
        print("Training error: ", np.mean(np.abs(scores["train_score"])))
        print("Validation error: ", np.mean(np.abs(scores["test_score"])))
    
    def mlp_regressor(self, preprocessor):
        one_hot_enc_data = pd.get_dummies(self.dev_X, columns=self.nominal_features)
        test_one_hot_enc = pd.get_dummies(self.test_X, columns=self.nominal_features)
        regr = MLPRegressor(random_state=1, max_iter=500, solver="adam").fit(one_hot_enc_data, 
                                                                             self.dev_y.ravel())
        
        # print(self.dev_X)
        # mlpRegressor = Pipeline([
        #     ("preprocessor", preprocessor),
        #     ("predictor", MLPRegressor())])
        # param_grid = [
        # {
        #     'predictor__activation' : ['identity', 'logistic', 'tanh', 'relu'],
        #     'predictor__solver' : ['adam'],
        #     'predictor__hidden_layer_sizes': [
        #      (15,),(16,),(17,)
        #      ], 
        #      'predictor__alpha': [1e-5],
        #      'predictor__max_iter': [200]}]
        # rlf = GridSearchCV(mlpRegressor, param_grid, cv=3,
        #                    scoring='accuracy')
        # rlf.fit(self.dev_X, self.dev_y)
        # print(rlf.best_params_)
        # validation_error = np.mean(cross_val_score(rlf, self.dev_X, self.dev_y, 
        #                                            scoring="neg_mean_absolute_error", cv=10))
        # print(f"mlp validation error: {validation_error}")


    def test_set(self, model):
        print(f"Mean absolute error on test data {mean_absolute_error(self.test_y,model.predict(self.test_X))}")

    def deploy(self, model, model_type):
        model.fit(self.df[self.training_features], self.df[self.target_feature].values)
        dump(model_type, 'models/my_model.pkl')

    
        


class RunAbalone:
    def __init__(self) -> None:
        self.abalone_df = pd.read_csv("datasets/abalone.csv") 
        self.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                        'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        self.nominal_features=['Sex']	
        self.numeric_features=['Length', 'Diameter', 'Height', 'Whole weight',
                               'Shucked weight', 'Viscera weight', 'Shell weight']
        self.abalone_df.columns = self.columns
        self.target_feature = ['Rings']
        self.training_features = [el for el in self.columns 
                                  if el not in self.target_feature]
        self.regression_abalone = RegressionModel(df=self.abalone_df, 
                                                  nominal_features=self.nominal_features, 
                                                  numeric_features=self.numeric_features, 
                                                  training_features=self.training_features, 
                                                  target_feature=self.target_feature)
        self.predictor_alpha = [0, 45.0]

    def run(self):
        print("Abalone age")
        self.regression_abalone.shuffle_df()
        self.regression_abalone.data_statistics()
        self.regression_abalone.split_training_test_data()
        preprocessor = self.regression_abalone.preprocessing(numeric_features=self.numeric_features, 
                                                             nominal_features=self.nominal_features)
        self.regression_abalone.prepare_for_model_selection()
        print("knn with dev data:")
        knn_gs, knn = self.regression_abalone.combine_preproc_with_knn(preprocessor=preprocessor,
                                               n_neighbours=[5, 6, 7, 8, 9, 10,
                                                                11, 12, 13])
        print("Ridge Regression:")
        self.regression_abalone.ridge_regression_pipeline(preprocessor=preprocessor, 
                                                                     predictor_alpha=self.predictor_alpha)
        print("knn with test data:")
        self.regression_abalone.test_set(knn_gs)
        self.regression_abalone.deploy(model=knn_gs, model_type=knn)
        self.regression_abalone.mlp_regressor(preprocessor=preprocessor)


class CPUStats:
    def __init__(self) -> None:
        self.df = pd.read_csv("datasets/cpu_hardware/machine.csv") 
        self.columns = ["vendor name", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", 
                        "CHMAX", "PRP", "ERP"]
        self.df.columns = self.columns
        self.nominal_features=["vendor name", "Model"]	
        self.numeric_features=["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", 
                        "CHMAX", "PRP"]
        self.target_feature = "ERP"
        self.training_features = [el for el in self.columns if el not in self.target_feature]
        print(self.training_features)
        self.regression_abalone = RegressionModel(df=self.df, 
                                                  nominal_features=self.nominal_features, 
                                                  numeric_features=self.numeric_features, 
                                                  training_features=self.training_features, 
                                                  target_feature=self.target_feature)
        self.predictor_alpha = [0, 45.0]

    def run(self):
        print("\n CPU performance")
        self.regression_abalone.shuffle_df()
        self.regression_abalone.data_statistics()
        self.regression_abalone.split_training_test_data()
        preprocessor = self.regression_abalone.preprocessing(numeric_features=self.numeric_features, 
                                                             nominal_features=self.nominal_features)
        self.regression_abalone.prepare_for_model_selection()
        print("knn with dev data:")
        knn_gs, knn = self.regression_abalone.combine_preproc_with_knn(preprocessor=preprocessor,
                                               n_neighbours=[6, 7, 8, 9, 10,
                                                                11, 12, 13])
        print("Ridge Regression:")
        self.regression_abalone.ridge_regression_pipeline(preprocessor=preprocessor, 
                                                                     predictor_alpha=self.predictor_alpha)
        print("knn with test data:")
        self.regression_abalone.test_set(knn_gs)
        self.regression_abalone.deploy(model=knn_gs, model_type=knn)
        # self.regression_abalone.mlp_regressor(preprocessor=preprocessor)

class WinesRegression:
    def __init__(self) -> None:
        self.df = pd.read_csv("datasets/scales/balance-scale.csv")
        self.columns = ['class name', 
                                  'Left-Weight', 
                                  'Left-Distance', 
                                  'Right-Weight', 
                                  'Right-Distance']
        self.df.columns = self.columns
        self.nominal_features=[]	
        self.numeric_features=['Left-Weight', 
                                  'Left-Distance', 
                                  'Right-Weight', 
                                  'Right-Distance']	
        self.target_feature = 'class name'
        self.training_features = [el for el in self.columns if el not in self.target_feature]
        print(self.training_features)
        self.regression_abalone = RegressionModel(df=self.df, 
                                                  nominal_features=self.nominal_features, 
                                                  numeric_features=self.numeric_features, 
                                                  training_features=self.training_features, 
                                                  target_feature=self.target_feature)
        self.predictor_alpha = [0, 45.0]

    def run(self):
        print("\n CPU performance")
        self.regression_abalone.shuffle_df()
        self.regression_abalone.data_statistics()
        self.regression_abalone.split_training_test_data()
        preprocessor = self.regression_abalone.preprocessing(numeric_features=self.numeric_features, 
                                                             nominal_features=self.nominal_features)
        self.regression_abalone.prepare_for_model_selection()
        print("knn with dev data:")
        knn_gs, knn = self.regression_abalone.combine_preproc_with_knn(preprocessor=preprocessor,
                                               n_neighbours=[6, 7, 8, 9, 10,
                                                                11, 12, 13])
        print("Ridge Regression:")
        self.regression_abalone.ridge_regression_pipeline(preprocessor=preprocessor, 
                                                                     predictor_alpha=self.predictor_alpha)
        print("knn with test data:")
        self.regression_abalone.test_set(knn_gs)
        self.regression_abalone.deploy(model=knn_gs, model_type=knn)
        # self.regression_abalone.mlp_regressor(preprocessor=preprocessor)

if __name__ == "__main__":
    RA = RunAbalone()
    RA.run()
    # RCPU = CPUStats()
    # RCPU.run()
    # WR = WinesRegression()
    # WR.run()





