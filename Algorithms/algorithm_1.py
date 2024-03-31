# symplectic_operations.py

import numpy as np
from numpy.fft import fftshift

class SymplecticOperations:
    def __init__(self):
        pass

    @staticmethod
    def gf2_gaussian_elimination(A, b):
        A, b = np.mod(A, 2), np.mod(b, 2)  # Ensure GF(2)
        m, n = A.shape
        Ab = np.hstack([A, b.reshape(-1, 1)])
        
        for col in range(n):
            for i in range(col, m):
                if Ab[i, col] == 1:
                    if i != col:
                        Ab[[col, i]] = Ab[[i, col]]
                    break
            else:
                continue
            
            for i in range(col + 1, m):
                if Ab[i, col] == 1:
                    Ab[i] = np.mod(Ab[i] + Ab[col], 2)
        
        x = np.zeros(n, dtype=int)
        for i in range(n - 1, -1, -1):
            if Ab[i, i] == 0 and Ab[i, -1] == 1:
                return None, False  # Inconsistent system
            if Ab[i, i] == 1:
                x[i] = np.mod(Ab[i, n] + np.dot(Ab[i, i+1:n], x[i+1:n]), 2)
        return x, True

    @staticmethod
    def Z_h(h, n):
        return np.mod(np.eye(2 * n) + np.mod(np.outer(fftshift(h), h), 2),2)
        
    @staticmethod
    def symp_inn_pdt(X, Y):
        return np.mod(np.sum(X * fftshift(Y, axes=1), axis=1), 2)

    def find_symp_mat_transvecs(self, X, Y):
        m, cols = X.shape
        n = cols // 2
        F = np.eye(2 * n)
        Transvecs = []
    
        for i in range(m):
            x_i, y_i = X[i, :], Y[i, :]
            x_it = np.mod(np.dot(x_i, F), 2)
            if np.all(x_i == y_i):
                continue
            if self.symp_inn_pdt(x_i.reshape(1, -1), y_i.reshape(1, -1)) == 1:
                #print(symp_inn_pdt(x_it.reshape(1, -1), y_i.reshape(1, -1)))
                h_i = np.mod(x_it + y_i, 2)
                #print(h_i)
                F = np.mod(np.matmul(F, self.Z_h(h_i, n)), 2)
                #print(x_it)
                #print(np.mod((x_it+y_i),2))
            
                #print(Z_h(h_i, n))
                Transvecs.append((self.Z_h(h_i, n), h_i))
            else:
                w_i = self.find_w(x_it, y_i, Y[:i, :], n)

                print(w_i)
                h_i1 = np.mod(w_i + y_i, 2)

                print(h_i1)
                h_i2 = np.mod(x_it + w_i, 2)
                print(h_i2)
                F = np.mod(np.dot(np.dot(F, self.Z_h(h_i1, n)), self.Z_h(h_i2, n)), 2)
                Transvecs.append((Z_h(h_i1, n), h_i1))
                Transvecs.append((Z_h(h_i2, n), h_i2))
        return F, Transvecs

    def find_w(self, x, y, Ys, n):
        A = fftshift(np.vstack([x, y] + [Ys[j, :] for j in range(Ys.shape[0])]), axes=1)
        b = np.array([1, 1] + [self.symp_inn_pdt(Ys[j, :].reshape(1, -1), y.reshape(1, -1))[0] for j in range(Ys.shape[0])])
        w, valid = self.gf2_gaussian_elimination(A, b)
        if not valid:
            raise ValueError("No valid solution found for w.")
        return w
