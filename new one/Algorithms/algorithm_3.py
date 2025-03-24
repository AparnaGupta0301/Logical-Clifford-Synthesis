import sys
import numpy as np
sys.path.append(r"C:\\Users\\gapar\\Logical-Clifford-Synthesis")
from Algorithms.algorithm_1 import SymplecticOperations


class SymplecticCodeSolver:
    def __init__(self):
        self.symplectic_ops = SymplecticOperations()

    def gf2matinv(self, matrix):
        """
        Compute the inverse of a matrix in GF(2) using Gaussian elimination.

        Parameters:
        matrix (numpy.ndarray): A square matrix to invert in GF(2).

        Returns:
        numpy.ndarray: The inverse of the matrix in GF(2).

        Raises:
        ValueError: If the matrix is singular (i.e., not invertible).
        """
        matrix = np.array(matrix, dtype=np.int8) % 2
        m = len(matrix)
        aug_matrix = np.concatenate((matrix, np.eye(m, dtype=np.int8)), axis=1)
        for col in range(m):
            for row in range(col, m):
                if aug_matrix[row, col]:
                    if row != col:
                        aug_matrix[[col, row]] = aug_matrix[[row, col]]
                    break
            else:
                raise ValueError("Matrix is singular over GF(2)")

            for i in range(m):
                if i != col and aug_matrix[i, col]:
                    aug_matrix[i] = (aug_matrix[i] + aug_matrix[col]) % 2

        return aug_matrix[:, m:]

    @staticmethod
    def bi2de(b):
        """
        Convert a binary matrix to a decimal.

        Parameters:
        b (numpy.ndarray): Binary matrix to convert.

        Returns:
        numpy.ndarray: Decimal representation of the binary matrix.
        """
        b = np.array(b)
        return b.dot(1 << np.arange(b.shape[-1])[::-1])

    def symplectic_code(self, U, V):
        """
        Generate all symplectic matrices satisfying the given constraints.
        
        This function solves the equation:
        U(1:2*m-k,:) * F = V
        
        Parameters:
        U (numpy.ndarray): A matrix forming a symplectic basis for F_2^(2m).
                           Must satisfy U*Omega*U' = Omega, where Omega = [0 I_m; I_m 0].
                           The rows of U are structured as [Xbar; S; Zbar; Spair], where 
                           Spair are vectors that complete the symplectic basis.
        V (numpy.ndarray): A matrix with (2m - k) rows that represents the stabilizer 
                           or logical conditions for the code.

        Returns:
        list: A list of symplectic matrices that satisfy the given constraints.
        """

        # m is the number of qubits, calculated from the number of columns of U (2m = number of columns)
        m = U.shape[1] // 2
        
        # k is the number of logical qubits, derived from the number of rows in V
        k = 2 * m - V.shape[0]
        
        # Calculate the total number of solutions, which is 2^(k*(k+1)//2)
        tot = 2 ** (k * (k + 1) // 2)
        F_all = [None] * tot

        # Find one particular solution using symplectic transvections
        F0, Transvecs = self.symplectic_ops.find_symp_mat_transvecs(U[0:(2*m-k), :], V)

        # Compute matrix A as U * F0 (mod 2) and its inverse in GF(2)
        A = U.dot(F0) % 2
        Ainv = self.gf2matinv(A)

        # Basis for the symplectic subspace is constructed from parts of A
        Basis = np.vstack((A[(m - k):m, :], A[(2*m - k):(2*m), :]))
        Subspace = np.mod(np.dot(np.array([list(np.binary_repr(i, 2*k)) for i in range(2**(2*k))], dtype=int), Basis), 2)

        # Stablizer generators in the subspace
        StabF0 = A[(m - k):m, :]
        Choices = [None] * k

        # Calculate all choices for each vector in the Subspace
        for i in range(k):
            h = np.concatenate((np.zeros(i, dtype=int), [1], np.zeros(k - i - 1, dtype=int)))
            Innpdts = np.dot(Subspace, np.fft.fftshift(StabF0, axes=1).T) % 2
            Choices[i] = Subspace[np.all(Innpdts == h, axis=1)]

        # Generate all symplectic matrices that satisfy the constraints
        for l in range(tot):
            Bl = np.copy(A)
            V_mat = np.zeros((k, 2 * m), dtype=int)
            lbin = np.array(list(np.binary_repr(l, width=k*(k+1)//2)), dtype=int)
            v1_ind = int(''.join(map(str, lbin[:k])), 2)
            V_mat[0, :] = Choices[0][v1_ind]

            # For each free vector, calculate corresponding choices
            for i in range(1, k):
                vi_ind = int(''.join(map(str, lbin[sum(range(k, k-i, -1)):(sum(range(k, k-i, -1))+k-i)])), 2)
                Innprods = np.dot(Choices[i], np.fft.fftshift(V_mat, axes=1).T) % 2
                Ch_i = Choices[i][np.all(Innprods == 0, axis=1)]
                V_mat[i, :] = Ch_i[vi_ind]

            # Construct Bl by placing V_mat into the last k rows
            Bl[(2 * m - k):(2 * m), :] = V_mat
            
            # Calculate the symplectic matrix F and store it in the list
            F = np.dot(Ainv, Bl) % 2
            F_all[l] = np.dot(F0, F) % 2

        return F_all
