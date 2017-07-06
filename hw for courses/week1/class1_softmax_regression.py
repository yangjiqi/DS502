import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt


# change the label to one hot vector
def onehot(y):
    n = len(np.unique(y))
    m = y.shape[0]
    b = np.zeros((m, n))
    for i in xrange(m):
        b[i,y[i]] = 1
    return b


#TODO softmax function
def softmax(X):
    return np.exp(X) / np.array([np.sum(np.exp(X), axis=1)]).T


# TODO hypothesis function
def h_func(theta, X):
    X = np.c_[np.ones(X.shape[0]), X]
    return softmax(np.dot(X, theta))


def h_gradient(theta, X, y, lam=0.1):
    n = X.shape[0]
    y_mat = onehot(y)
    preds = h_func(theta, X) # 1797 x 10

    # TODO compute gradient

    mul = np.dot(np.c_[np.ones(X.shape[0]), X].T, (y_mat - preds))
    grad = mul * (-1.0) / n
    # grad = np.sum(mul, axis = 0) / n
    grad += lam * np.r_[np.array([np.zeros(theta.shape[1])]), theta[1:, :]]
    return grad



def softmax_cost_func(theta, X, y, lam=0.1):
    n = X.shape[0]
    y_mat = onehot(y)

    # TODO compute loss (cost)
    probs = h_func(theta, X)
    mul = y_mat * np.log(probs)
    cost = np.sum(mul) * (-1.0) / n
    reg_cost = lam * (np.sum(theta * theta) - np.sum(theta[0,:] * theta[0,:])) * 0.5
    cost += reg_cost
    return cost


# gradient descent
def softmax_grad_desc(theta, X, y, lr=.01, converge_change=.0001, max_iter=100, lam=0.1):
    # normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    cost_iter = []
    cost = softmax_cost_func(theta, X, y, lam=lam)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while change_cost > converge_change and i < max_iter:
        pre_cost = cost
        theta -= lr * h_gradient(theta, X, y)
        cost = softmax_cost_func(theta, X, y)
        cost_iter.append([i, cost])
        change_cost = abs(pre_cost - cost)
        i += 1
    return theta, np.array(cost_iter)


def softmax_pred_val(theta, X):
    probs = h_func(theta, X)
    preds = np.argmax(probs, axis=1)
    return probs, preds


def softmax_regression():
    # Load the diabetes dataset
    dataset = datasets.load_digits()

    # Use all the features
    X = dataset.data[:, :]
    y = dataset.target[:, None]

    # Gradiend Descent
    theta = np.random.rand(X.shape[1]+1, len(np.unique(y)))
    fitted_val, cost_iter = softmax_grad_desc(theta, X, y, lr=0.01, max_iter=1000, lam=0.1)
    probs, preds = softmax_pred_val(fitted_val, X)

    #print(fitted_val)
    print(cost_iter[-1,:])
    print('Accuracy: {}'.format(np.mean(preds[:, None] == y)))

    plt.plot(cost_iter[:, 0], cost_iter[:, 1])
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()


def main():
    softmax_regression()

if __name__ == "__main__":
    main()