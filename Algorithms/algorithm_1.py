# symplectic_operations.py

import numpy as np
from numpy.fft import fftshift
from helper_functions.helperfunctions import *

class SymplecticOperations:
    def __init__(self):
        pass

    def find_symp_mat_transvecs(self, X, Y):
        m, cols = X.shape
        n = cols // 2
        F = np.eye(2 * n)
        Transvecs = []
    
        for i in range(m):
            x_i, y_i = X[i, :], Y[i, :]
            x_it = (np.matmul(x_i, F))%2 
            if np.all(x_it == y_i):
               
                continue
            if self.symp_inn_pdt(x_it.reshape(1, -1), y_i.reshape(1, -1)) == 1:
                
                h_i = np.mod(x_it + y_i, 2)
                
                F = np.mod(np.matmul(F, self.Z_h(h_i, n)), 2)
                
                Transvecs.append((self.Z_h(h_i, n), h_i))
            else:
                
                w_i = self.find_w(x_it, y_i, Y[:i, :], n)            
        
                h_i1 = np.mod(w_i + y_i, 2)
                
                h_i2 = np.mod(x_it + w_i, 2)
                
                F = np.mod(np.dot(np.dot(F, self.Z_h(h_i1, n)), self.Z_h(h_i2, n)), 2)
                
                Transvecs.append((self.Z_h(h_i1, n), h_i1))
                Transvecs.append((self.Z_h(h_i2, n), h_i2))
        return F, Transvecs

    def find_w(self, x, y, Ys, n):
        A = fftshift(np.vstack([x, y] + [Ys[j, :] for j in range(Ys.shape[0])]), axes=1)
       # A = np.fft.fftshift(np.vstack([x, y, Ys]), axes=1)
        b = np.array([1, 1] + [self.symp_inn_pdt(Ys[j, :].reshape(1, -1), y.reshape(1, -1))[0] for j in range(Ys.shape[0])])
        w, valid = gf2_gaussian_elimination_with_echelon(A, b)
        if not valid:
            raise ValueError("No valid solution found for w.")
        return w
    
    def Z_h(self, h, n):
        return np.mod(np.eye(2 * n) + np.mod(np.outer(fftshift(h), h), 2),2)
        
    def symp_inn_pdt(self, X, Y):
        return np.mod(np.sum(X * fftshift(Y, axes=1), axis=1), 2)
