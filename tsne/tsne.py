import numpy as np
import pylab as Plot

def pca(X,n_dims = 50):
    (n, d) = X.shape
    X = X - np.mean(X, 0)
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:,0:n_dims])
    return Y

def pairwise_L2(X):
    sum_X = np.sum(X * X, axis = 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return D

def xdist_to_p(D,perplexity = 30.0,tol = 1e-5):
    n = D.shape[0]
    p_out = np.zeros((n,n))
    target_entropy = np.log(perplexity)
    # tune precision for each point's gaussian distribution
    for i in range(n):
        precision_min = -np.inf
        precision_max = np.inf
        precision = 1.0
        flag = False
        max_try = 50

        loop = 0
        while (not flag):
            ds = D[i]
            ps = np.exp(-ds * precision)
            ps[i] = 0

            #normalization
            s = np.sum(ps)
            ps /= s
            # hack for log(0)
            ps[i] = 1
            entropy = np.sum(-ps * np.log(ps))
            # recover 0 value in p[i,i]
            ps[i] = 0

            if (entropy > target_entropy):
                precision_min = precision
                precision = 2.0 * precision if precision_max == np.inf else (precision + precision_max) / 2.0
            else:
                precision_max = precision
                precision = precision / 2.0 if precision_min == -np.inf else (precision + precision_min) / 2.0
            loop += 1
            if (np.abs(entropy - target_entropy) < tol or loop >= max_try):
                flag = True

        p_out[i] = ps

    p_out = (p_out + p_out.T) / (2 * n)
    index = p_out < 1e-100
    p_out[index] = 1e-100
    return p_out

def ydist_to_q(D):
    n = D.shape[0]
    q_out = 1.0 / (1.0 + D)
    q_out[range(n),range(n)] = 0.0
    q_out /= np.sum(q_out)
    index = q_out < 1e-100
    q_out[index] = 1e-100
    return q_out


def tsne(X,perplexity = 30.0,n_dims = 2,max_iter = 1000,min_gain = 0.01,learning_rate = 100):
    n = X.shape[0]
    Dx = pairwise_L2(X)
    Px = xdist_to_p(Dx,perplexity = 30.0,tol = 1e-5)

    Y = np.random.randn(n, n_dims)
    iY = np.zeros((n, n_dims))

    gains = np.ones((n, n_dims))


    iter_num = 0
    while iter_num < max_iter:
        iter_num += 1

        Dy = pairwise_L2(Y)
        Ty = 1.0 / (1.0 + Dy)
        Qy = ydist_to_q(Dy)

        pmul = 4.0 if iter_num < 100 else 1.0
        momentum = 0.5 if iter_num < 20 else 0.8

        grads = np.zeros(Y.shape)
        for i in range(n):
            grad = 4 * np.dot((pmul * Px[i] - Qy[i]) * Ty[i],Y[i] - Y)
            grads[i] = grad

        gains = (gains + 0.2) * ((grads > 0) != (iY > 0)) + (gains * 0.8) * ((grads > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain;
        iY = momentum * iY - learning_rate * (gains * grads);
        Y = Y + iY
        Y = Y - np.mean(Y, 0)

        # Compute current value of cost function
        if (iter_num + 1) % 10 == 0:
            C = np.sum(Px * np.log(Px / Qy))
            print "Iteration ", (iter_num + 1), ": error is ", C
    return Y


if __name__ == '__main__':
    X = np.loadtxt("mnist2500_X.txt");
    labels = np.loadtxt("mnist2500_labels.txt")
    X = pca(X, 50).real
    Y = tsne(X)
    Plot.scatter(Y[:,0], Y[:,1], 20, labels)
    Plot.show()















