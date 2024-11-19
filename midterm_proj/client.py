from utils.preprocessing import load_and_clean_data, convert_to_numpy
from sklearn.model_selection import train_test_split
from utils.evaluation import Evaluation
from models.LinearRegression import LinearRegression, threshold_classifier

def main():
    X, y = load_and_clean_data(ovveride=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = convert_to_numpy(X_train, X_test, y_train, y_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    eval = Evaluation()


    # Linear Regression
    lr = LinearRegression(n_feature=8)
    lr.MBGD(X_train, y_train)
    y_pred = lr.predict(X_test)
    linear_prediction = threshold_classifier(y_pred)

    eval.evaluate(y_test, linear_prediction, positive_label=1)

    print()



if __name__ == '__main__':
    main()


