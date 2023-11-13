import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
import pandas as pd
from . import methods
class Svm():
    def __init__(self,C):
        self.C = C
        self.w = None
        self.b = None
        self.kernel_name = None

    def fit(self, X, y):
        # preprocess:
        X = np.array(X)
        y = np.array(y)
        N, d = X.shape
        # Obtain Quadratic Programming
        P, q, G, h, A, b = methods.dual_problem_quadratic_program(X, y, self.C)

        #Solve Quadratic Program
        sol = methods.dual_problem_quadratic_solver(P, q,G, h, A, b)

        # Caculate Lagrange 
        lam = methods.svm_lagrange_mutipliers(sol)

        # Find Svm suport vectors that lam > 0
        S = methods.svm_support_vectors(lam)

        # Find weight
        self.w = methods.svm_weight(X, y, lam)
        # Find bias 
        self.b = methods.svm_bias(X, y, S, self.w)
        self.w = np.array(self.w)
        self.b = np.array(self.b)

    def predict(self, X):
        X2 = np.array(X)
        H = np.sign(X2.dot(self.w)+self.b)
        return H

    def decision_function(self, X):
        if self.kernel_name == 'linear':
            return X.dot(self.w) + self.b
        else:
            N = X.shape[0]
            y_predict = np.zeros(N)
            # dataset is a support vector
            X_sv = self.X[self.S]
            y_sv = self.y[self.S]
            # compute 
            for i in range (N):
                y_predict[i] = np.sum(self.lam[self.S] * y_sv * self.kernel.compute(X_sv, X[i]))
            return y_predict + self.b