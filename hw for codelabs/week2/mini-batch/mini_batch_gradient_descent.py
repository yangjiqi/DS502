from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn

def data_preprocessing(shuffle_data = False):
    # read source data from csv file
    filename = 'pima-indians-diabetes.data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data_frame = read_csv(filename, names=names)
    array = data_frame.values

    # Separate array into input and output components
    X = array[:, 0:8]
    y = array[:, 8]
    # change y shape of (n,) to (n, 1)
    y = np.array([y]).T

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    # Summarize transformed data
    np.set_printoptions(precision=3)

    if shuffle_data == True:
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation, :]
        y = y[permutation]
    # X, y = Shuffle(X, y)

    return X, y


def split_data(X, y, train_percentage = 0.7):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percentage, random_state=29)
    return X_train, X_test, y_train, y_test


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def logistic_val_func(theta, x):
    # forwarding
    # return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], theta.T))
    return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], theta.T))

def logistic_cost_func(theta, x, y):
    # compute cost (loss)
    y_hat = logistic_val_func(theta, x)
    cost = np.sum(y * np.log(y_hat)) + np.sum((1 - y) * np.log(1-y_hat))
    cost *= 1.0 / x.shape[0]
    return -cost


def logistic_grad_func(theta, x, y):
    # compute gradient
    grad = np.dot((logistic_val_func(theta, x) - y).T, np.c_[np.ones(x.shape[0]), x])
    grad = grad / x.shape[0]

    return grad

def mini_batch_gradient_descent(theta, X_train, y_train, lr=0.00001, epochs=1000,
                                  batch_size=50,
                                  momentum_flag = False,
                                  momentum=0.9,
                                  epsilon=1e-20,
                                  verbose = False,
                                  ):
    m = len(X_train)
    error = np.zeros(theta.shape[1])

    error_list = []
    cost = logistic_cost_func(theta, X_train, y_train)
    error_list.append(cost)

    for epoch in range(epochs):
        prev_theta = 0
        for batch_i in range(m // batch_size):
            batch_X = X_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            batch_y = y_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            # print(batch_i + 1)
            grad = logistic_grad_func(theta, batch_X, batch_y)
            if momentum_flag:
                theta -= (lr * (grad * 1.0 / batch_size) - momentum * prev_theta)
                prev_theta = lr * (grad * 1.0 / batch_size) - momentum * prev_theta
            else:
                theta -= lr * (grad * 1.0 / batch_size)

            # print("batch cost ", logistic_cost_func(theta, batch_X, batch_y))

        cost = logistic_cost_func(theta, X_train, y_train)
        error_list.append(cost)
        if verbose:
            print ("epoch:{0} cost:{1}".format(epoch, error_list[-1]))

        if np.linalg.norm(theta - error) < epsilon:
            break
        else:
            error = theta

    return theta, error_list


def pred_val(theta, X, hard=True):
    # prediction values
    # normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_val_func(theta, X)
    pred_value = np.where(pred_prob > 0.5, 1, 0)
    if hard:
        return pred_value
    else:
        return pred_prob

def accuracy(theta, X_test, y_test):
    correct = np.sum((pred_val(theta, X_test) == y_test))
    return correct * 1.0 / X_test.shape[0]

def f1_score(X_train, X_test, y_train, y_test):
    cls = linear_model.LogisticRegression()

    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    return cm, f1


if __name__ == '__main__':
    # simple mini-batch
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = split_data(X, y)

    theta1 = np.random.rand(1, X_train.shape[1] + 1)
    theta1, error_list = mini_batch_gradient_descent(theta1, X_train, y_train)
    # cm1, f1_1 = f1_score(X_train, X_test, y_train, y_test)
    print("Accuracy of mini-batch : {}".format(accuracy(theta1, X_test, y_test)))
    # print("metrics:\n {}".format(cm1))
    # print("f1 score: {}".format(f1_1))

    theta2 = np.random.rand(1, X_train.shape[1] + 1)
    theta2, error_list = mini_batch_gradient_descent(theta2, X_train, y_train, momentum_flag=True)
    print("Accuracy of mini-batch with momentum(0.9) : {}".format(accuracy(theta2, X_test, y_test)))
