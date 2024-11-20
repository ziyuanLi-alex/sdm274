import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import minmax_scale, StandardScaler
import os
from sklearn.utils import resample
import pandas as pd
from imblearn.over_sampling import SMOTE

def load_and_clean_data(normalize=True, ovveride=False):
    ###
    # Load and clean the data
    # param controls normalization of the data
    # returns: X, y_trunc
    ###

    # Check if processed files exist
    X_path = '/root/sdm274/midterm_proj/data/X.csv'
    y_path = '/root/sdm274/midterm_proj/data/y.csv'
    # normalize = False

    if os.path.exists(X_path) and os.path.exists(y_path) and not ovveride:
        X = pd.read_csv(X_path)
        y_trunc = pd.read_csv(y_path)
        return X, y_trunc
    
    # No missing values in the data  
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601)
    X = ai4i_2020_predictive_maintenance_dataset.data.features
    y_raw = ai4i_2020_predictive_maintenance_dataset.data.targets
    y_trunc = y_raw.iloc[:, 0]
    # X.iloc[:, 1:] = X.iloc[:, 1:].astype(float)

    # Note that the data is mixed
    # numerical columns are also mixed with int64 and float64
    categorical_columns = ["Type"]
    numerical_columns = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]

    # Convert the categorical columns to one-hot encoding
    X = pd.get_dummies(X, columns=categorical_columns)
    # Convert all boolean columns to integers (0 and 1)
    X[['Type_H', 'Type_L', 'Type_M']] = X[['Type_H', 'Type_L', 'Type_M']].astype(float)

    # for linear regression, we should drop the third column afterwards.

    # Normalize the data
    if normalize:
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    return X, y_trunc

def convert_to_numpy(X_train, X_test, y_train, y_test):
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def undersample_data(X, y, random_state=42):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # Extract the first column if it's a DataFrame
    
    target_column = 'target'
    data = pd.concat([X, y.rename(target_column)], axis=1)
    
    majority_class = data[data[target_column] == 0]
    minority_class = data[data[target_column] == 1]
    
    majority_undersampled = resample(majority_class,
                                   replace=False,
                                   n_samples=len(minority_class),
                                   random_state=random_state)
    
    # Combine minority class with undersampled majority class
    balanced_data = pd.concat([majority_undersampled, minority_class])
    
    # Shuffle the data
    balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Separate features and target
    X_reduced = balanced_data.drop(columns=[target_column])
    y_reduced = balanced_data[target_column]
    
    # print("\nBalanced distribution:")
    # print(f"Majority class (0): {len(y_reduced[y_reduced == 0])}")
    # print(f"Minority class (1): {len(y_reduced[y_reduced == 1])}")
    
    return X_reduced, y_reduced

def enhance_data(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def undersample_data_numpy(X, y, random_state=42):
    """
    Perform undersampling on a dataset represented as NumPy arrays.
    
    Parameters:
        X (np.ndarray): Feature dataset of shape (n_samples, n_features).
        y (np.ndarray): Target dataset of shape (n_samples,).
        random_state (int): Random seed for reproducibility.
    
    Returns:
        X_reduced (np.ndarray): Undersampled feature dataset.
        y_reduced (np.ndarray): Undersampled target dataset.
    """
    # Ensure y is 1D
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.flatten()

    # Separate majority and minority classes
    majority_class_idx = np.where(y == 0)[0]
    minority_class_idx = np.where(y == 1)[0]

    # Undersample majority class
    undersampled_majority_idx = resample(
        majority_class_idx,
        replace=False,
        n_samples=len(minority_class_idx),
        random_state=random_state
    )

    # Combine minority class and undersampled majority class
    balanced_idx = np.concatenate([undersampled_majority_idx, minority_class_idx])
    np.random.seed(random_state)
    np.random.shuffle(balanced_idx)  # Shuffle the indices

    # Create the reduced datasets
    X_reduced = X[balanced_idx]
    y_reduced = y[balanced_idx]

    return X_reduced, y_reduced



if __name__ == "__main__":
    X, y = load_and_clean_data()
    X_balanced, y_balanced = undersample_data(X, y)
    # X.to_csv('/root/sdm274/midterm_proj/data/X.csv', index=False)
    # y.to_csv('/root/sdm274/midterm_proj/data/y.csv', index=False)

