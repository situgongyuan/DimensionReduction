import numpy as np

"""
pca by eigen decomposition
"""
def eig_pca(X,n_dims):
    # X: matrix of (N,D),N is the total number of samples, D is the dimension of features
    (N,D) = X.shape
    mean = np.mean(X, 0)
    X = X - mean
    cov = np.dot(X.T, X) / N
    (l, M) = np.linalg.eig(cov)
    Y = np.dot(X, M[:,0:n_dims])
    return Y

"""
pca by singular value decomposition
"""
def svd_pca(X,n_dims):
    U,Sigma,Vt = np.linalg.svd(X)
    V = Vt.T
    Y = np.dot(X,V[:,0:n_dims])
    return Y

"""
pca by power iteration
"""
def power_pca(X,n_dims,max_iter=100):
    (N,D) = X.shape
    mean = np.mean(X, 0)
    X = X - mean
    cov = np.dot(X.T, X) / N
    eigen_vectors = np.zeros(D,n_dims)
    for i in range(n_dims):
        p = np.random.rand(1,D)
        p = p / np.sqrt(np.sum(p ** 2))
        p = p.reshape(D,1)
        for j in range(max_iter):
            q = np.dot(cov,p)
            p = q / np.sqrt(np.sum(p ** 2))
        eigen_vectors[:,i] = p
        eigen_value = np.dot(np.dot(p.T,cov),p)
        cov = cov - eigen_value * np.dot(p,p.T)
    Y = np.dot(X,eigen_vectors)
    return Y





