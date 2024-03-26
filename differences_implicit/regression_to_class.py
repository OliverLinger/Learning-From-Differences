# regression_to_classification.py

from collections import Counter
import numpy as np


class RegressionToClassificationConverter:
    def __init__(self, n_segments=3, equal_division=None):
        """
        Initialize RegressionToClassificationConverter.

        Parameters:
        - n_segments (int): Number of segments to divide the continuous target variable into.
        - equal_division (bool): If True, divide the data into segments with nearly equal counts.
        """
        self.n_segments = n_segments
        self.unique_ranges = None
        self.equal_division = equal_division

    def fit(self, X, y=None):
        """
        Fit the model.

        Parameters:
        - X (array-like): Input features.
        - y (array-like): Target variable. Not used.

        Returns:
        - self: Returns an instance of self.
        """
        # For transformers, fit() method doesn't necessarily do anything
        return self
    
    def transform(self, y, min_val=None, max_val=None, unique_ranges=None):
        """
        Convert a regression problem into a classification problem by segmenting the continuous target variable y
        into n_segments distinct segments.

        Parameters:
        - y (list or array-like): The continuous target variable.
        - min_val (float): Minimum value of the target variable.
        - max_val (float): Maximum value of the target variable.
        - unique_ranges (array-like): Predefined unique ranges for segments.

        Returns:
        - categories (list): A list of categories (segment indices) corresponding to each value in y.
        - unique_ranges (dict or array-like): Unique ranges for segments.
        - min_val (float): Minimum value of the target variable.
        - max_val (float): Maximum value of the target variable.
        """
        # Check if this is the predict phase
        if min_val is None and max_val is None:
            min_val = min(y)
            max_val = max(y)

        # Whether we are dividing based on max and min, or placing items equally in n classes
        if self.equal_division is None:
            # Calculate the range for each segment
            segment_range = (max_val - min_val) / self.n_segments

            # Initialize an empty list to store the categories
            categories = []
            # Initialize an empty dictionary to store the unique string ranges and their corresponding segment index
            unique_ranges = {}
            current_index = 0
            # Iterate through each value in y
            for val in y:
                # Determine the range for the current value
                for i in range(self.n_segments):
                    if i < self.n_segments - 1:
                        if min_val + i * segment_range <= val < min_val + (i + 1) * segment_range:
                            category = f"{int(min_val + i * segment_range)}-{int(min_val + (i + 1) * segment_range)}"
                            if category not in unique_ranges:
                                unique_ranges[category] = current_index
                                current_index += 1
                            categories.append(unique_ranges[category])
                            break
                    else:  # Last segment
                        if min_val + i * segment_range <= val <= max_val:
                            category = f"{int(min_val + i * segment_range)}-{int(max_val)}"
                            if category not in unique_ranges:
                                unique_ranges[category] = current_index
                                current_index += 1
                            categories.append(unique_ranges[category])
                            break
            return categories, unique_ranges, min_val, max_val
        else:
            # Divide the data into n_segments based on nearly equal counts
            quantiles = np.linspace(0, 1, self.n_segments + 1)
            if unique_ranges is None:
                unique_ranges = np.quantile(y, quantiles)
            categories = np.digitize(y, unique_ranges[:-1])
            category_counts = Counter(categories)
            print("Category Counts:", category_counts)
            return categories, unique_ranges, min_val, max_val


    def inverse_transform(self, categories):
        """
        Inverse transform the categories back to continuous values if needed.

        Parameters:
        - categories (array-like): The categories to be inverse-transformed.

        Returns:
        - None: This method does not return any value.
        """
        # Implement inverse transformation if needed
        pass
