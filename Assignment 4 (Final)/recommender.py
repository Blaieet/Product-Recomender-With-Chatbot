import os
import numpy as np
import pandas as pd
import pickle

from scipy.stats.stats import pearsonr
from numpy import linalg as LA

try:
    from tqdm import tqdm_notebook as tqdmn
except:
    tqdmn = lambda x, leave=True: x
    
    
def matrix_factorization(X,P,Q,K,steps,alpha,beta):
    nonnull = np.where(~np.isnan(X))
    
    Q = Q.T
    for step in tqdmn(range(steps)):
        for idx in tqdmn(range(nonnull[0].size), leave=False):
            i, j = nonnull[0][idx], nonnull[1][idx]
    
            #calculate the error of the element
            eij = X[i][j] - np.dot(P[i,:],Q[:,j])
            
            #second norm of P and Q for regularilization
            sum_of_norms = LA.norm(P) + LA.norm(Q)
            
            #print sum_of_norms
            eij += ((beta / 2.0) * sum_of_norms)
            
            #compute the gradient from the error
            P[i, :] += alpha * (2 * eij * Q[:, j] - (beta * P[i, :]))
            Q[:, j] += alpha * (2 * eij * P[i, :] - (beta * Q[:, j]))
            
        V = P.dot(Q)
        error = np.sum(np.power(X[nonnull] - V[nonnull], 2))
        
        if error < 0.001:
            break
            
    return P, Q.T


def factorize(X, K, steps=5000):
    N = X.shape[0]
    M = X.shape[1]
    #P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
    P = np.random.rand(N, K)
    #Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
    Q = np.random.rand(M, K)
    
    alpha = 0.0002
    beta = float(0.02)
    
    P, Q = matrix_factorization(X, P, Q, K, steps, alpha, beta)
    return P, Q


class NMFRecommender():
    def __init__(self, data_train, num_features, num_steps):
        self.df = data_train
        self.num_features = num_features
        self.num_steps = num_steps
        self.idmap = dict(zip(self.df.index, range(self.df.index.size)))

    def factorize(self):
        try:
            with open('P.pkl', 'rb') as fp: self.P = pickle.load(fp)
            with open('Q.pkl', 'rb') as fp: self.Q = pickle.load(fp)
        except:
            self.P, self.Q = factorize(self.df.values, self.num_features, self.num_steps)
            
            with open('P.pkl', 'wb') as fp: pickle.dump(self.P, fp)
            with open('Q.pkl', 'wb') as fp: pickle.dump(self.Q, fp)
        
    def estimate(self, user, item):
        return self.P[self.idmap[user],:].dot(self.Q.T[:,item - 1])