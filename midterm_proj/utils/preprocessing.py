from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import minmax_scale, StandardScaler
import os
import pandas as pd

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



if __name__ == "__main__":
    X, y = load_and_clean_data()
    X.to_csv('/root/sdm274/midterm_proj/data/X.csv', index=False)
    y.to_csv('/root/sdm274/midterm_proj/data/y.csv', index=False)

