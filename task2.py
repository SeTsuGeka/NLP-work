import numpy as np
from scipy.stats import norm

def em(X,max_iter,k=2):
    n = X.shape[0]
    mu=[180,165]
    sig = np.ones(k) * X.std()
    w = np.ones(k) / k
    for m in range(max_iter):
        # E-step
        prob = np.array([norm.pdf(X, loc=mu[i], scale=sig[i]) for i in range(k)]).T
        gamma = prob * w
        gamma /= gamma.sum(axis=1)[:, None]
        # M-step
        w = gamma.mean(axis=0)
        mu = (gamma * X[:, None]).sum(axis=0) / gamma.sum(axis=0)
        sig = np.sqrt((gamma * (X[:, None] - mu) ** 2).sum(axis=0) / gamma.sum(axis=0))
        # if w[0] == w[0]:
        #     break
        print('第{}次迭代的结果为，男生有{}人，男生身高的均值为{}，标准差为{}，女生有{}人，女生身高的均值为{}，标准差为{}'.format(m + 1,int(w[0]*n),mu[0],sig[0],int(w[1]*n)+1,mu[1],sig[1]))

    return mu, sig, w

if __name__ == '__main__':
    data = np.loadtxt(open('C:/Users/zzy/Desktop/课程相关/001_NLP/第二次作业/DLNLP2023-main/height_data.csv'), delimiter=',',
                      skiprows=1, usecols=0)
    mu,sig,w=em(data,200,k=2)
    n=data.shape[0]
    print('最终结果为，男生有{}人，男生身高的均值为{}，标准差为{}，女生有{}人，女生身高的均值为{}，标准差为{}'.format(int(w[0] * n), mu[0], sig[0],
                                                                                int(w[1] * n) + 1, mu[1], sig[1]))
