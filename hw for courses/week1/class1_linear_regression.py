import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import time


def plot_line(x, y, y_hat,line_color='blue'):
    # Plot outputs
    plt.scatter(x, y,  color='black')
    plt.plot(x, y_hat, color=line_color,
             linewidth=3)
    plt.xticks(())
    plt.yticks(())

    plt.show()


def linear_grad_func(theta, x, y):
    # TODO compute gradient
    # m = x.shape[0]
    # h = linear_val_func(theta, x)
    # a1 = h - y
    # a2 = np.c_[np.ones(x.shape[0]), x]
    # temp_multiply = a1 * a2
    #
    # grad = np.sum(temp_multiply, axis = 0) / m
    m = y.size
    y_hat = linear_val_func(theta, x)
    x = np.c_[np.ones(x.shape[0]), x]
    grad = np.sum((y_hat - y) * x, axis = 0) * 1.0 / m
    # grad = np.sum((linear_val_func(theta, x) - y) * np.c_[np.ones(x.shape[0]), x], axis = 0) * 1.0 / x.shape[0]
    return grad


def linear_val_func(theta, x):
    # forwarding
    return np.dot(np.c_[np.ones(x.shape[0]), x], theta.T)  # return a column


def linear_cost_func(theta, x, y):
    # TODO compute cost (loss)
    m = x.shape[0]
    h = linear_val_func(theta, x)
    cost = np.sum((h - y) ** 2) / (2 * m)
    return cost


def linear_grad_desc(theta, X_train, Y_train, lr=0.1, max_iter=10000, converge_change=.001):

    cost_iter = []
    cost = linear_cost_func(theta, X_train, Y_train)
    cost_iter.append([0, cost])
    cost_change = 1
    i = 1
    while cost_change > converge_change and i< max_iter:
        pre_cost = cost
        # compute gradient
        grad = linear_grad_func(theta, X_train, Y_train)

        # TODO Update gradient
        theta = theta - lr * grad


        cost = linear_cost_func(theta, X_train, Y_train)
        cost_iter.append([i, cost])
        cost_change = abs(cost - pre_cost)
        i += 1

    return theta, cost_iter


def linear_regression():
    # load dataset
    dataset = datasets.load_diabetes()
    # Select only 2 dims
    X = dataset.data[:, 2]
    Y = dataset.target

    # split dataset into training and testing
    X_train = X[:-20, None]
    X_test = X[-20:, None]

    Y_train = Y[:-20, None]
    Y_test = Y[-20:, None]


    # Linear regression
    theta = np.random.rand(1, X_train.shape[1]+1)  # init, 1 x (shape[1] + 1) , 1 x 2, a row
    fitted_theta, cost_iter = linear_grad_desc(theta, X_train, Y_train, lr=0.1, max_iter=50000)

    print('Coefficients: {}'.format(fitted_theta[0,-1]))
    print('Intercept: {}'.format(fitted_theta[0,-2]))
    print('MSE: {}'.format(np.sum((linear_val_func(fitted_theta, X_test) - Y_test)**2) / Y_test.shape[0]))
    # plt.figure(1)

    plot_line(X_test, Y_test, linear_val_func(fitted_theta, X_test))
    plt.draw()
    time.sleep(5)
    plt.close(1)


def sklearn_linear_regression():
    # load dataset
    dataset = datasets.load_diabetes()
    # Select only 2 dims
    X = dataset.data[:, 2]
    Y = dataset.target

    # split dataset into training and testing
    X_train = X[:-20, None]
    X_test = X[-20:, None]

    Y_train = Y[:-20, None] # last 20 lines
    Y_test = Y[-20:, None]

    # Linear regression
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, Y_train)
    print('Coefficients: {}'.format(regressor.coef_))
    print('Intercept: {}'.format(regressor.intercept_))
    print('MSE:{}'.format(np.mean((regressor.predict(X_test) - Y_test) ** 2)))

    plt.figure(2)
    plot_line(X_test, Y_test, regressor.predict(X_test),line_color='red')
    plt.show()


def main():
    print('Class 1 Linear Regression Example')
    linear_regression()

    print ('')

    print('sklearn Linear Regression Example')
    sklearn_linear_regression()


if __name__ == "__main__":
    main()