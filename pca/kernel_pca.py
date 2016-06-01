import numpy as np

def rbf_kernel(X,Y,sigma):
    XY = np.dot(X,Y.T)
    XX = np.sum(X ** 2,axis = 1,keepdims = True)
    YY = np.sum(Y ** 2,axis = 1)
    D = XX - 2 * XY + YY
    K = np.exp(D / (-2 * sigma ** 2))
    return K


def linear_kernel(X,Y):
    return np.dot(X,Y.T)

def centering(K):
    N = K.shape[0]
    s = np.sum(K) / (N ** 2)
    ones = np.ones((N,N))
    K = K - 1 / N * np.dot(K,ones) - 1 / N * np.dot(ones,K) + s
    return K

def kernel_pca(X,sigma,n_dims):
    K = rbf_kernel(X,X,sigma)
    K = centering(K)
    (l, M) = np.linalg.eig(K)
    Y = np.dot(K,M[:,0:n_dims])
    return Y
