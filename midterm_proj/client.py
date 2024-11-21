
from matplotlib import pyplot as plt
from utils.preprocessing import *
from sklearn.model_selection import train_test_split
from utils.evaluation import Evaluation
from models.LinearRegression import LinearRegression, threshold_classifier
from sklearn.preprocessing import MinMaxScaler
from models.Perceptron import Perceptron
from models.LogisticRegression import LogisticRegression
from models.MLP import MLPClassifier

def main():
    X, y = load_and_clean_data(ovveride=False)
    # X, y = undersample_data(X, y)
    # X, y = enhance_data(X, y)

    models = {'Linear Regression': False, 'Perceptron': False, 'Logistic Regression': False, 'MLP': False}

    if models['Linear Regression']:
        X_lin = X.drop(columns=['Type_M'])
        X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y, test_size=0.3, random_state=42)
        X_train_lin, X_test_lin, y_train_lin, y_test_lin = convert_to_numpy(X_train_lin, X_test_lin, y_train_lin, y_test_lin)
        X_train_lin, y_train_lin = enhance_data(X_train_lin, y_train_lin)
        X_test_lin, y_test_lin = undersample_data_numpy(X_test_lin, y_test_lin)

        # print(X_train_lin.shape, X_test_lin.shape, y_train_lin.shape, y_test_lin.shape)
        eval = Evaluation()

        # Linear Regression
        lr = LinearRegression(n_feature=7, lr=1e-6, epochs=500)
        lr.BGD(X_train_lin, y_train_lin)
        y_pred = lr.predict(X_test_lin)
        # Min-Max Regularization for y_pred
        scaler = MinMaxScaler()
        y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()


        linear_prediction = threshold_classifier(y_pred, threshold=0.5)
        print("Linear Regression")
        eval.evaluate(y_test_lin, linear_prediction, positive_label=1)
        print()

        # eval.plot_confusion_matrix(y_test_lin, linear_prediction)
        # eval.ROC(y_test_lin, y_pred)
        # Plot the loss
        # plt.plot(lr.loss)
        # plt.show()

    if models['Perceptron']:
        X_train_per, X_test_per, y_train_per, y_test_per = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_per, X_test_per, y_train_per, y_test_per = convert_to_numpy(X_train_per, X_test_per, y_train_per, y_test_per)
        X_train_per, X_test_per, y_train_per, y_test_per = enhance_undersamp(X_train_per, X_test_per, y_train_per, y_test_per)
        # print(X_train_per.shape, X_test_per.shape, y_train_per.shape, y_test_per.shape)
        eval = Evaluation()
        y_train_per = map_y(y_train_per)
        # y_test_per = map_y(y_test_per)
        mapper = lambda y: -1 if y == 1 else 1
        y_test_per = np.array([mapper(yi) for yi in y_test_per])

        # Perceptron
        per = Perceptron(n_feature=8, lr=1e-7, epochs=500)
        per.BGD(X_train_per, y_train_per)
        y_pred = per.predict(X_test_per)
        # Min-Max Regularization for y_pred
        # scaler = MinMaxScaler()
        # y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

        # eval.ROC(y_test_per, y_pred)
        per_prediction = threshold_classifier(y_pred, threshold=0.5)
        print("Perceptron")
        eval.evaluate(y_test_per, per_prediction, positive_label=1)
        print()


        # Plot the loss
        # plt.plot(per.loss)
        # plt.show()

    if models['Logistic Regression']:
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_log, X_test_log, y_train_log, y_test_log = convert_to_numpy(X_train_log, X_test_log, y_train_log, y_test_log)
        X_train_log, X_test_log, y_train_log, y_test_log = enhance_undersamp(X_train_log, X_test_log, y_train_log, y_test_log)
        # print(X_train_log.shape, X_test_log.shape, y_train_log.shape, y_test_log
        eval = Evaluation()
        y_train_log = map_y(y_train_log)
        y_test_log = map_y(y_test_log)
        mapper = lambda y: 0 if y == 1 else 1
        y_test_log = np.array([mapper(yi) for yi in y_test_log])

        # Logistic Regression
        log = LogisticRegression(n_feature=8, learning_rate=0.6, epochs=500)
        log.BGD(X_train_log, y_train_log)
        y_pred = log.predict(X_test_log)
        # Min-Max Regularization for y_pred
        scaler = MinMaxScaler()
        y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

        # eval.ROC(y_test_per, y_pred)

        print("Logistic Regression")
        eval.evaluate(y_test_log, y_pred, positive_label=1)
        print()

        # Plot the loss
        # plt.plot(log.loss)
        # plt.show()

    if models['MLP']:
        X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = convert_to_numpy(X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp)
        # print(X_train_mlp.shape, X_test_mlp.shape, y_train_mlp.shape, y_test_mlp.shape)
        eval = Evaluation()

        


if __name__ == '__main__':
    main()


