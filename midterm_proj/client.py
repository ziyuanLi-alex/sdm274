
from matplotlib import pyplot as plt
from utils.preprocessing import *
from sklearn.model_selection import train_test_split
from utils.evaluation import Evaluation
from models.LinearRegression import LinearRegression, threshold_classifier
from sklearn.preprocessing import MinMaxScaler
from models.Perceptron import Perceptron
from models.LogisticRegression import LogisticRegression
from models.MLP import Linear, MLPClassifier, ReLU, Sigmoid

normalization = True

def main():
    X, y = load_and_clean_data(normalize=normalization)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = convert_to_numpy(X_train, X_test, y_train, y_test)
    # X_train, X_test, y_train, y_test = enhance_undersamp(X_train, X_test, y_train, y_test)
    X_train, y_train = enhance_data(X_train, y_train)
    X_test, y_test = undersample_data_numpy(X_test, y_test)

    eval = Evaluation()


    models = {'Linear Regression': False, 'Perceptron': True, 'Logistic Regression': False, 'MLP': False}

    if models['Linear Regression']:
        # Dropping the last column
        X_train_lin = X_train[:, :-1]
        X_test_lin = X_test[:, :-1]
        # print(X_train_lin.shape, X_test_lin.shape, y_train_lin.shape, y_test_lin.shape)
        eval = Evaluation()

        # Linear Regression
        lr = LinearRegression(n_feature=7, lr=1e-7, epochs=500)
        lr.BGD(X_train_lin, y_train)
        y_pred = lr.predict(X_test_lin)
        # Min-Max Regularization for y_pred
        scaler = MinMaxScaler()
        y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()


        linear_prediction = threshold_classifier(y_pred, threshold=0.5)
        print("Linear Regression", f'Normalized: {normalization}')
        eval.evaluate(y_test, linear_prediction, positive_label=1)
        print()

        # eval.plot_confusion_matrix(y_test_lin, linear_prediction)
        # eval.ROC(y_test, y_pred)
        # Plot the loss
        # plt.plot(lr.loss)
        # plt.show()

    if models['Perceptron']:
        # print(X_train_per.shape, X_test_per.shape, y_train_per.shape, y_test_per.shape)
        eval = Evaluation()
        y_train_per = map_y(y_train)
        # y_test_per = map_y(y_test_per)
        mapper = lambda y: -1 if y == 1 else 1
        y_test_per = np.array([mapper(yi) for yi in y_test])

        # Perceptron
        per = Perceptron(n_feature=8, lr=1e-6, epochs=500)
        per.BGD(X_train, y_train_per)
        y_pred = per.predict(X_test)
        # Min-Max Regularization for y_pred
        # scaler = MinMaxScaler()
        # y_pred = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

        # eval.ROC(y_test_per, y_pred)
        print("Perceptron", f'- Normalized: {normalization}')
        eval.evaluate(y_test_per, y_pred, positive_label=-1)
        print()


        # Plot the loss
        # plt.plot(per.loss)
        # plt.show()

        # Plot ROC
        # eval.ROC(y_actual=y_test_per, y_pred=y_pred)

    if models['Logistic Regression']:
        # X is ranged data with many features.
        # y_train is the target data, coded into 1 and 2.
        # y_mapped is the true data, coded into 0 and 1. original data 1 is endoced into 0 and 2 is encoded into 1.
        
        # use the original y_train(1 and 2) to train the model, and use the y_mapped( 1 -> 0, 2 -> 1) to evaluate the model.

        # In this case, 1 represents failure
        eval = Evaluation()
        train_mapper = lambda y: 1 if y == 0 else 2
        y_train_log = np.array([train_mapper(yi) for yi in y_train])

        # mapper = lambda y: 0 if y == 2 else 1
        # y_test_log = np.array([mapper(yi) for yi in y_test])

        # Logistic Regression
        log = LogisticRegression(n_feature=8, learning_rate=0.6, epochs=500)
        log.BGD(X_train, y_train_log)
        y_pred = log.predict(X_test)

        print("Logistic Regression")
        eval.evaluate(y_test, y_pred, positive_label=1)
        print()

        # Plot the loss
        # plt.plot(log.loss)
        # plt.show()

        # eval.ROC(y_actual=y_test_log, y_pred=y_pred)

    if models['MLP']:
        # print(X_train_mlp.shape, X_test_mlp.shape, y_train_mlp.shape, y_test_mlp.shape)
        layers = [
        Linear(input_size=8, output_size=64),
        ReLU(),
        Linear(input_size=64, output_size=128),
        ReLU(),
        # Linear(input_size=128, output_size=256),
        # ReLU(),
        Linear(input_size=128, output_size=1),
        Sigmoid()
        ]   

        epochs = 500
        lr = 1e-5

        mlp = MLPClassifier(layers, epochs=epochs, lr=lr, input_shape=X_train.shape[1])
        mlp.reset()
        mlp.train_MBGD(X_train, y_train)


        y_pred = mlp.predict(X_test)
        eval.evaluate(y_test, y_pred, positive_label=1)

        loss_history = mlp.loss
        length = len(loss_history)

        # plt.figure(figsize=(10, 6))
        # plt.plot(range(length), loss_history, label='Training Loss')
        # plt.xlabel('Batches')
        # plt.ylabel('Loss')
        # plt.title('Training Loss over Batches')
        # plt.legend()
        # plt.grid(True)
        # plt.show()




if __name__ == '__main__':
    main()


