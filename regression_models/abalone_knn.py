import pandas as pd
from pandas.plotting import scatter_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from typing import List
# I create a labelled data set so that the machine learning model will carry out supervised learning
# Get the dataset as a dataframe.


class RegressionAbalone:
    def __init__(self, abalone_df: pd.DataFrame, nominal_features: List, numeric_features: List) -> None:
        self.df = abalone_df
        self.dev_df = pd.DataFrame
        self.tes_df = pd.DataFrame
        self.nominal_features = nominal_features 
        self.numeric_features = numeric_features 

    def shuffle_df(self) -> None: 
        # Shuffle the dataset
        self.df = self.df.sample(frac=1, random_state=2)
        self.df.reset_index(drop=True, inplace=True)

    
    def data_statistics(self) -> None:
        print(self.df.describe(include="all"))


    def split_training_test_data(self) -> None:
        self.dev_df, self.test_df = train_test_split(self.df, train_size=0.8, random_state=2)


    def nominal_numerical_details(self) -> None:
        # This is for nominal features and there are no null values, No editing is needed
        for feature in self.nominal_features:
            print(feature, self.dev_df[feature].unique())
        for feature in self.numeric_features:
            print(feature, self.dev_df[feature].unique()) 

    def describe_data_visually(self) -> None:
        # This plot illustrates the correlation between the diameter of a snail and the number of rings it has.
        sns.scatterplot( x="Rings", y="Diameter", data=self.dev_df)
        # Visualise the data so to look for deviations 
        # First I will use a copy of the dataframe 
        copy_df = self.dev_df.copy()
        numeric_abalone_df = abalone_df[self.numeric_features]
        sns.heatmap(numeric_abalone_df.corr(), annot=True)
        scatter_matrix(copy_df, figsize=(15, 15))
        plt.show()
 

    ''' dealing with data '''
    def remove_outliers(self)-> None:
        # There seems to be outliers in the height catagory
        #get the indexes of the height outliers
        index_of_height_outlier = self.dev_df.Height[self.dev_df.Height>0.4].index
        # There are 2 outliers, drop them inplace so that a pandas series isnt created and it actually edits the dataframe inplace
        self.dev_df.drop(index_of_height_outlier, inplace=True)
        # There seems to be outliers in the height catagory


    def convert_catagorical_to_nominal(self) -> None:
        # Using one hot encoding to express the nominal data as a binary expression
        # I am using one hot encoding snce there is no ordered relationship between the option in the sex column 
        one_hot_enc_data = pd.get_dummies(self.dev_df, columns=['Sex'])
        self.dev_df = one_hot_enc_data

    def preprocessor(self, numeric_features: List):
        # Create the preprocessor
        preprocessor = ColumnTransformer([
            ("scaler", StandardScaler(), numeric_features),
            ("nom", Pipeline([("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")), 
                ("binarizer", OneHotEncoder(handle_unknown="ignore"))]), 
                nominal_features)], remainder="passthrough")
        return preprocessor
    


# #Feature seperation 
# features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight','Shucked weight', 'Viscera weight', 'Shell weight']
# target = ['Rings']
# x_train = dev_abalone_df[features]
# y_trian = dev_abalone_df[target]
"""---------------------------------------------------------"""

if __name__ == "__main__":
    abalone_df = pd.read_csv("datasets/abalone.csv")	
    abalone_df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight','Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    nominal_features=['Sex']	
    numeric_features=['Length', 'Diameter', 'Height', 'Whole weight','Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    regression_abalone = RegressionAbalone(abalone_df=abalone_df, nominal_features=nominal_features, numeric_features=numeric_features)
    regression_abalone.shuffle_df()
    regression_abalone.data_statistics()
    regression_abalone.split_training_test_data()
    #regression_abalone.nominal_numerical_details()
    #regression_abalone.describe_data_visually()
    regression_abalone.remove_outliers()
    regression_abalone.convert_catagorical_to_nominal()