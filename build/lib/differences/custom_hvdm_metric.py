import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def vdm(a, b, feature_index, class_labels, data):
    """
    Calculates the Value Difference Metric (VDM) for a categorical feature.

    Parameters:
    - a: The value of the feature for instance a.
    - b: The value of the feature for instance b.
    - feature_index: The index of the feature in the dataset.
    - class_labels: The list of all class labels in the dataset.
    - data: The dataset, where the last column is assumed to be the class label.

    Returns:
    - The VDM distance between the two feature values a and b.
    """
    # Length of the dataset
    n = len(data)
    # Initialize sum for VDM calculation
    vdm_sum = 0
    # Iterate over each class label to calculate part of the VDM sum
    for c in class_labels:
        # Count instances where feature value is a and class label is c
        n_a_c = sum((data[:, feature_index] == a) & (data[:, -1] == c))
        # Count instances where feature value is b and class label is c
        n_b_c = sum((data[:, feature_index] == b) & (data[:, -1] == c))
        # Count instances where feature value is a
        n_a = sum(data[:, feature_index] == a)
        # Count instances where feature value is b
        n_b = sum(data[:, feature_index] == b)
        # Calculate part of VDM sum for class c and add to total sum
        vdm_sum += (abs(n_a_c/n_a - n_b_c/n_b) if n_a > 0 and n_b > 0 else 1) ** 2
    # Return the square root of the VDM sum as the final distance
    return np.sqrt(vdm_sum)

def hvdm(a, b, numeric_indices, categorical_indices, class_labels, data, scaler):
    """
    Calculates the Heterogeneous Value Difference Metric (HVDM) by combining 
    the distances for numeric and categorical features.

    Parameters:
    - a, b: Instances between which the distance is to be calculated.
    - numeric_indices: List of indices for numeric features in the dataset.
    - categorical_indices: List of indices for categorical features in the dataset.
    - class_labels: The list of all class labels in the dataset.
    - data: The dataset, where the last column is assumed to be the class label.
    - scaler: A fitted StandardScaler instance for scaling numeric features.

    Returns:
    - The HVDM distance between instances a and b.
    """
    # Initialize the distance
    distance = 0
    # Calculate the distance for numeric features
    for index in numeric_indices:
        # Scale and compute difference for numeric features
        diff = scaler.transform([[a[index]]])[0][0] - scaler.transform([[b[index]]])[0][0]
        # Add squared difference to total distance
        distance += diff ** 2
    # Calculate the distance for categorical features using VDM
    for index in categorical_indices:
        # Add squared VDM distance for each categorical feature to total distance
        distance += vdm(a[index], b[index], index, class_labels, data) ** 2
    # Return the square root of the total distance as the final HVDM distance
    return np.sqrt(distance)

def compute_distance_matrix(X, hvdm_metric):
    """Computes the HVDM distance matrix for a dataset.
    
    Parameters:
    - X: DataFrame, the dataset for which to compute the distance matrix.
    - hvdm_metric: callable, the HVDM metric function that computes the distance between two instances.

    Returns:
    - A NumPy array representing the distance matrix.
    """
    num_samples = len(X)
    distance_matrix = np.zeros((num_samples, num_samples))
    
    # Fetch necessary data for the HVDM calculation
    data = X.to_numpy()
    class_labels = np.unique(data[:, -1])  # Assuming the last column is the class label
    scaler = StandardScaler().fit(X[numeric_features].to_numpy())  # Re-fit scaler for numeric features
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):  # Matrix is symmetric
            distance = hvdm(data[i], data[j], numeric_indices, categorical_indices, class_labels, data, scaler)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            
    return distance_matrix

df = pd.read_csv("datasets/abalone/abalone.csv")
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
           'Rings']
features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
numeric_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
nominal_features = ['Sex']

df.columns = columns
dev_X = df[features]
dev_y = df["Rings"]

numeric_indices = [features.index(feature) for feature in numeric_features]
categorical_indices = [features.index(feature) for feature in nominal_features]

# Assuming your class labels are in a separate variable `y` and part of the DataFrame `X`
class_labels = dev_y.unique()

# StandardScaler should be fit on numeric features of your training data

# Now, you can compute the distance matrix
distance_matrix = compute_distance_matrix(dev_X, hvdm)