import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def drop_one_type(X):
    P = X.drop(columns=['Type_M'])
    return P
    

def evaluate_correlation(X):
    # Evaluate the correlation between features by calculating a correlation matrix
    # and then plotting it
    correlation_matrix = X.corr()
    print(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def drop_air_temperature(X):
    # Drop the air temperature column
    P = X.drop(columns=['Air temperature'])
    return P

def evaluate_significance(X, y):
    # Evaluate the significance of the features by calculating the correlation between
    # each feature and the target variable
    y_series = y.iloc[:, 0]  # Get first column as Series
    correlations = X.corrwith(y_series)
    print(correlations)

    sns.barplot(x=correlations.index, y=correlations)
    plt.title("Correlation with target variable")
    plt.show()



if __name__ == "__main__":
    X = pd.read_csv('/root/sdm274/midterm_proj/data/X.csv')
    y = pd.read_csv('/root/sdm274/midterm_proj/data/y.csv')
    evaluate_correlation(X)
    # evaluate_significance(X, y)




