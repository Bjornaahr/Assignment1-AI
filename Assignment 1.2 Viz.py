# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

#%%
data = np.loadtxt(open("X:\Skule\AI\EX1\ex1data1.txt", "r"), delimiter=",")
X = data[:, 0]
y = data[:, 1]
m = len(y)  # Number of training examples
print("Input", X)
print("Output", y)
print("Number of Datapoints", m)


#%%
def plot_data(x, y):
    """
    Plots the data points x and y.

    Parameters
    ----------
    x : array-like
        Data on x axis.
    y : array-like
        Data on y axis.
    """
    plt.plot(x, y, linestyle='', marker='*', color='c', label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')


#%%
plt.figure()
plot_data(X, y)
plt.show()


#%%
n = len(X)
print(n)
print(X.shape)


#%%
X = np.hstack((np.ones((m, 1)), X.reshape(m, 1)))
print("Shape of X for matrix multiplication", X.shape)
print("Shape of Y", y.shape)


#%%
theta = np.zeros(2)
print(theta)


#%%
iterations = 1700
alpha = 0.001


#%%
def compute_cost(x, y, theta):
    """
    Compute cost for linear regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    theta : ndarray, shape (n_features,)
        Linear regression parameter.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in X and y.
    """
    m = len(y)
    J = np.sum(np.square(x.dot(theta) - y)) / (2.0 * m)

    return J


#%%
cost = compute_cost(X, y, theta)
print ('The cost on initial theta:', cost)


#%%
def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    theta : ndarray, shape (n_features,)
        Initial linear regression parameter.
    alpha : float
        Learning rate.
    num_iters: int
        Number of iteration.

    Returns
    -------
    theta : ndarray, shape (n_features,)
        Linear regression parameter.
    J_history: ndarray, shape (num_iters,)
        Cost history.
    """
    m = len(y)
    J_history = np.zeros(num_iters)
    theta_History = np.zeros([num_iters, 2])

    for i in range(num_iters):
        theta -= alpha / m * ((X.dot(theta) - y).T.dot(X))
        J_history[i] = compute_cost(X, y, theta)
        theta_History[i] = theta


    return theta, J_history, theta_History

#%%
def f(e, d):
       return e**2 + d**2    

#%%
theta, cost, theta_History = gradient_descent(X, y, theta, alpha, iterations)

d, e = map(list, zip(*theta_History))

fig = plt.figure()
ax = fig.gca(projection='3d')


a, b = np.meshgrid(e, d)
e, d = np.meshgrid(cost, cost)
c = f(e, d)

surf = ax.plot_surface(a,b,c, cmap=cm.coolwarm)
ax.set_xlabel("b0")
ax.set_zlabel("cost")
ax.set_ylabel('b1')



plt.show()

#%%
print ("Theta found by gradient descent: \n", theta)


#%%
plt.figure()
plot_data(X[:, 1], y)
plt.plot(X[:, 1], X.dot(theta), label='Linear Regression')
plt.legend(loc='upper left', numpoints=1)
plt.show()


#%%
predict1 = np.array([1, 3.5]).dot(theta)
print ("For population = 35,000, we predict a profit of", predict1 * 10000)

predict2 = np.array([1, 7]).dot(theta)
print ("For population = 70,000, we predict a profit of", predict2 * 10000)

predict3 = np.array([1, 9]).dot(theta)
print ("For population = 90,000, we predict a profit of", predict3 * 10000)

predict4 = np.array([1, 22]).dot(theta)
print ("For population = 220,000, we predict a profit of", predict4 * 10000)


#%%
