# node level kernel
import numpy as np
import math
from numpy.random import multivariate_normal
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def simulate_expectation(cov, activate='relu', repeat=50):
    def relu(x):
        return 0 if x < 0 else x

    def indicat(x):
        return 0 if x <= 0 else 1

    x, y = multivariate_normal([0, 0], cov, repeat).T
    func = relu if activate == 'relu' else indicat
    return np.mean([func(x[i]) * func(y[i]) for i in range(repeat)])
"""
def exp(cov, ep=1e-10):
    '''
    if cov[0, 1] ** 2 == cov[0, 0] * cov[1, 1]:
        return 0.5 * math.sqrt(cov[0, 1] ** 2), 0.5
    '''
    v_x = cov[0, 0]
    v_y = cov[1, 1]
    rho = cov[0, 1] / math.sqrt(v_x * v_y)
    if rho >= 1:
        return 0.5 * math.sqrt(v_x * v_y), 0.5
    rho_ = 1 - rho ** 2
    A = 2 * math.pi * math.sqrt(v_x * v_y * rho_)
    a = - 1 / (2 * rho_ * v_x + ep)
    b = rho / (rho_ * math.sqrt(v_x * v_y) + ep)
    if b<0:
        print(b)
    c = - 1 / (2 * rho_ * v_y + ep)
    delta = 4 * a * c - b ** 2
    #20220417
    '''
    relu = - (b ** 2 - 2 * a * c) / (2 * a * c * delta + ep) + \
        b * math.sqrt(delta) * np.arctan(math.sqrt(delta / (b ** 2 + ep))) \
         / (delta ** 2 + ep)
    relu /= (A + ep)
    
    '''
    relu = - (b ** 2 + b * abs(b) - 4 * a * c) / (4 * a * c * delta + ep) + \
        b * math.sqrt(delta) * np.arctan(math.sqrt(delta / (b ** 2 + ep)))/ (delta ** 2 + ep)  
    relu /= (A + ep)
    
    #20220417
    indi = np.arctan(math.sqrt(delta / (b ** 2 + ep))) / math.sqrt(delta + ep)
    indi /= (A + ep)
    
    if str(relu) == 'nan' or str(indi) == 'nan':
        raise(1)
    
    return relu, indi
"""
#no ep (works)
def exp(cov):
    '''
    if cov[0, 1] ** 2 == cov[0, 0] * cov[1, 1]:
        return 0.5 * math.sqrt(cov[0, 1] ** 2), 0.5
    '''
    v_x = cov[0, 0]
    v_y = cov[1, 1]
    rho = cov[0, 1] / math.sqrt(v_x * v_y)
    if rho >= 1:
        return 0.5 * math.sqrt(v_x * v_y), 0.5
    rho_ = 1 - rho ** 2
    A = 2 * math.pi * math.sqrt(v_x * v_y * rho_)
    a = - 1 / (2 * rho_ * v_x)
    b = rho / (rho_ * math.sqrt(v_x * v_y))
    c = - 1 / (2 * rho_ * v_y )
    delta = 4 * a * c - b ** 2
    #20220417
    '''
    relu = - (b ** 2 - 2 * a * c) / (2 * a * c * delta + ep) + \
        b * math.sqrt(delta) * np.arctan(math.sqrt(delta / (b ** 2 + ep))) \
         / (delta ** 2 + ep)
    '''
    relu = - (b ** 2 + b * abs(b) - 4 * a * c) / (4 * a * c * delta) + \
        b * math.sqrt(delta) * np.arctan(math.sqrt(delta / (b ** 2)))/ (delta ** 2)  
    relu /= (A)
    
    #20220417
    indi = np.arctan(math.sqrt(delta / (b ** 2))) / math.sqrt(delta)
    indi /= (A)
    if str(relu) == 'nan' or str(indi) == 'nan':
        raise(1)
    
    return relu, indi

cov = np.array([[1,0],[0,1]])
print(exp(cov))
print(exp(np.array([[1,1],[1,1]])))
exit()


def nodeNTK(H, A, L=3, R=2, c_sigma=2, decay = 0.2):  #decay 0420
    '''
    input: 
        H: n_node * n_feature np array, initial vector for each node
        A: adjacency matrix, 0 or 1, no self loop
        l: # block
        R: deepth for each block
    output: n*n matrix representing the nodeNTK
    '''
    n, _ = H.shape
    '''
    d = np.sum(A, 1).reshape((-1, 1))
    c = np.multiply(1/(d+1), 1/(d+1).T)   
    '''
    c = 1. #1. / np.array(np.sum(A, axis=1) * np.sum(A, axis=0))
    #(0)(R)
    s = H.dot(H.T)
    t = s.copy()

                
    '''          
    s = np.multiply((np.identity(n) + A).dot(s).dot(np.identity(n) + A), c)  
    t = np.multiply((np.identity(n) + A).dot(t).dot(np.identity(n) + A), c) 
    '''
    
    def aggregate(A,m):  #m can be s(sigma) or t(theta)  #n is the dimension of m
        print('aggregating')
        final = np.zeros((n, n))
        A_new = A+np.identity(n)
        for i in tqdm(range(n)):  
            for j in range(n):
                ind1 = np.where(A_new[i]==1)[0]
                ind2 = np.where(A_new[j]==1)[0]
                final[i][j] = np.sum(m[ind1][:,ind2])
        
        final = np.multiply(final, c)  
        return final
    
    print('start')
    
    #s = aggregate(A,s)  #20220419 aggf
    #t = aggregate(A,t)
    
    for l in range(1,L+1):  #range(L+1)
        print('Block ',l)
        s = aggregate(A,s)   #should be
        t = aggregate(A,t)      
        for r in range(R):
            s_new = np.zeros((n, n))
            s_dot = np.zeros((n, n))
            for i in tqdm(range(n)):
                for j in range(n):
                    if j < i: 
                        s_new[i, j] = s_new[j, i]
                        s_dot[i, j] = s_dot[j, i]
                    else:
                        a = s[[i, j], :][:, [i, j]]
                        try:
                            relu, indi = exp(a)
                        except:
                            print(a, i, j)
                            raise(1)
                        s_new[i, j] = c_sigma * relu
                        s_dot[i, j] = c_sigma * indi
            s = s_new
            t = np.multiply(t, s_dot) + s
            np.savez('s_'+str(l)+'_'+str(r), t)
            np.savez('cache_'+str(l)+'_'+str(r), t)
        #s = aggregate(A,s)  #20220419  aggf
        #t = aggregate(A,t)
        
        c_sigma = c_sigma*decay  #20220420
        '''
        s = np.multiply((np.identity(n) + A).dot(s).dot(np.identity(n) + A), c)  
        t = np.multiply((np.identity(n) + A).dot(t).dot(np.identity(n) + A), c)  
        '''
    # s = aggregate(A,s)  #wagg
    # t = aggregate(A,t)

    return t




'''
from data_loader import load_data

features, labels, adj, train_mask, val_mask, test_mask = load_data('cora')

H = np.zeros(features[-1])
for [i, j], k in zip(features[0], features[1]):
     H[i, j] = k

A = np.zeros(adj[-1])
for [i, j], k in zip(adj[0], adj[1]):
    if i != j:
        A[i, j] = k

K = nodeNTK(H, A)
'''
