# symplectic_operations.py

import numpy as np
from numpy.fft import fftshift

class SymplecticOperations:
    def __init__(self):
        pass

    @staticmethod
    def reduce_to_echelon_form(A, b):
        """Reduce the matrix A to echelon form over GF(2) and apply the same operations to b."""
        rows, cols = A.shape
        A = np.copy(A) % 2
        b = np.copy(b) % 2
        row = 0

        for col in range(cols):
            if row >= rows:
                break
        
            # Find a pivot row for this column
            pivot = np.where(A[row:, col] == 1)[0]
            if len(pivot) == 0:
                continue  # No pivot in this column, move to the next column
            pivot_row = pivot[0] + row
            
            # Swap the current row with the pivot row
            A[[row, pivot_row]] = A[[pivot_row, row]]
            b[[row, pivot_row]] = b[[pivot_row, row]]
            
            # Eliminate all other ones in this column
            for r in range(rows):
                if r != row and A[r, col] == 1:
                    A[r] = (A[r] + A[row]) % 2  # Modified this line
                    b[r] = (b[r] + b[row]) % 2  # Modified this line
            row += 1
        return A, b

    @staticmethod
    def solve_from_echelon_form(A, b):
        """Solve the system given A in echelon form."""
        rows, cols = A.shape
        x = np.zeros(cols, dtype=int)
        
        for r in range(rows - 1, -1, -1):
            if np.any(A[r]) == 0:
                continue  # Skip rows that are all zeros
            col = np.argmax(A[r])  # Find the first 1 in this row
            x[col] = (b[r] - np.dot(A[r], x)) % 2
        return x

    def gf2_gaussian_elimination_with_echelon(self, A, b):
        A_echelon, b_echelon = self.reduce_to_echelon_form(A, b)
        return self.solve_from_echelon_form(A_echelon, b_echelon), True

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
            #print("x_i")
            #print(x_i)
           # print("y_i")
           # print(y_i)
            x_it = np.mod(np.matmul(x_i, F), 2)
            if np.all(x_it == y_i):
               
                continue
            if self.symp_inn_pdt(x_i.reshape(1, -1), y_i.reshape(1, -1)) == 1:
                
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
        
        b = np.array([1, 1] + [self.symp_inn_pdt(Ys[j, :].reshape(1, -1), y.reshape(1, -1))[0] for j in range(Ys.shape[0])])
        
        w, valid = self.gf2_gaussian_elimination_with_echelon(A, b)
        if not valid:
            raise ValueError("No valid solution found for w.")
        return w
