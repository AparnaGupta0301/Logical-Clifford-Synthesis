import numpy as np
import sys

# Import necessary modules
sys.path.append(r"C:\Users\gapar\Logical-Clifford-Synthesis")
from Algorithms.algorithm_1 import SymplecticOperations
from helper_functions.helperfunctions import SymplecticMatrix

# Initialize the SymplecticOperations class
symplectic_ops = SymplecticOperations()

class FindAllSympMat:
    """
    Class to handle symplectic code solving and finding symplectic matrices that satisfy the equation:
    U([I, m+J], :) * F = V.
    
    The rows of U must form a symplectic basis for F_2^(2m), meaning U must satisfy:
    U * Omega * U' = Omega, where Omega = [0 I_m; I_m 0].
    """
    def __init__(self):
        self.symplectic_ops = SymplecticOperations()
    
    @staticmethod
    def gf2matinv(matrix):
        """
        Compute the inverse of a matrix in GF(2) using Gaussian elimination.

        Parameters:
        matrix (numpy.ndarray): A square matrix to invert in GF(2).

        Returns:
        numpy.ndarray: The inverse of the matrix in GF(2).

        Raises:
        ValueError: If the matrix is singular (i.e., not invertible).
        """
        matrix = np.array(matrix, dtype=np.int8) % 2  # Ensure GF(2) operations
        m = len(matrix)
        aug_matrix = np.concatenate((matrix, np.eye(m, dtype=np.int8)), axis=1)  # Augment with identity matrix

        # Gaussian elimination for GF(2)
        for col in range(m):
            for row in range(col, m):
                if aug_matrix[row, col]:
                    if row != col:
                        aug_matrix[[col, row]] = aug_matrix[[row, col]]  # Swap rows
                    break
            else:
                raise ValueError("Matrix is singular over GF(2)")

            for i in range(m):
                if i != col and aug_matrix[i, col]:
                    aug_matrix[i] = (aug_matrix[i] + aug_matrix[col]) % 2  # Row operation

        return aug_matrix[:, m:]  # Return the inverse portion

    @staticmethod
    def intersect(arr1, arr2):
        """
        Find the intersection between two arrays and return the indices.

        Parameters:
        arr1 (numpy.ndarray): First array.
        arr2 (numpy.ndarray): Second array.

        Returns:
        numpy.ndarray: Indices of intersection elements.
        """
        common_elements, ind_arr1, ind_arr2 = np.intersect1d(arr1, arr2, return_indices=True)
        return ind_arr1

    @staticmethod
    def setdiff(arr1, arr2):
        """
        Mimics MATLAB's setdiff, returning elements in arr1 not in arr2.

        Parameters:
        arr1 (numpy.ndarray): First array.
        arr2 (numpy.ndarray): Second array.

        Returns:
        numpy.ndarray: Array of elements present in arr1 but not in arr2.
        """
        return np.setdiff1d(arr1, arr2)

    def find_all_symp_mat(self, U, V, I, J):
        """
        Find all symplectic matrices F that satisfy U([I,m+J],:) * F = V.
        U must form a symplectic basis for F_2^(2m), i.e., U*Omega*U' = Omega.

        Parameters:
        U (numpy.ndarray): Matrix forming a symplectic basis.
        V (numpy.ndarray): Matrix with constraints.
        I (list): Indices for part of the symplectic equation.
        J (list): Indices for part of the symplectic equation.

        Returns:
        list: A list of symplectic matrices F that satisfy the constraints.
        """
        m = U.shape[1] // 2  # Calculate m from the number of columns of U
        Omega = np.block([[np.zeros((m, m)), np.eye(m)], [np.eye(m), np.zeros((m, m))]])

        # Verify if U is a symplectic matrix
        if not np.all((U @ Omega @ U.T) % 2 == Omega):
            print('\nInvalid matrix U in function find_all_symp_mat\n')
            return [None]

        # Convert input indices to arrays
        I = np.array(I).flatten()
        J = np.array(J).flatten()

        # Compute the complement of I and J in [1, m]
        Ibar = self.setdiff(np.arange(1, m+1), I)
        Jbar = self.setdiff(np.arange(1, m+1), J)
        alpha = len(Ibar) + len(Jbar)

        # Total number of solutions
        tot = 2 ** (alpha * (alpha + 1) // 2)
        F_all = [None] * tot

        # Find one solution using symplectic transvections
        F0, *_ = self.symplectic_ops.find_symp_mat_transvecs(U[np.ix_(np.concatenate([I-1, m+J-1]), )], V)

        # Calculate matrix A and its inverse
        A = (U @ F0) % 2
        Ainv = self.gf2matinv(A)

        # Combine Ibar and Jbar
        IbJb = np.union1d(Ibar, Jbar)

        # Compute the basis for subspace
        Basis = A[np.ix_(np.concatenate([IbJb-1, m+IbJb-1]), )]
        Subspace = (np.array([list(format(i, f'0{2*len(IbJb)}b')) for i in range(2**(2*len(IbJb)))]).astype(int) @ Basis) % 2

        # Find indices of fixed basis vectors
        Basis_fixed_I = self.intersect(IbJb, I)
        Basis_fixed_J = self.intersect(IbJb, J)
        Basis_fixed = Basis_fixed_I
        Basis_free = np.setdiff1d(np.arange(2*len(IbJb)), Basis_fixed)

        # Choices for each vector in subspace
        Choices = [None] * alpha

        # Calculate choices for free vectors
        for i in range(alpha):
            ind = Basis_free[i]
            h = np.zeros(len(Basis_fixed))
            if i < len(Ibar):
                h[Basis_fixed == len(IbJb) + ind] = 1
            else:
                h[Basis_fixed == ind - len(IbJb)] = 1

            Innpdts = (Subspace @ np.fft.fftshift(Basis[Basis_fixed, :], axes=1).T) % 2
            Choices[i] = Subspace[np.all(np.mod(Innpdts, 2) == h, axis=1), :]

        # Generate all symplectic matrices
        for l in range(tot):
            Bl = A.copy()
            W = np.zeros((alpha, 2*m))

            # Convert l to binary
            lbin = format(l, f'0{alpha*(alpha+1)//2}b')

            # Process free vectors
            v1_ind = int(lbin[:alpha], 2)
            W[0, :] = Choices[0][v1_ind, :]
            for i in range(1, alpha):
                vi_ind = int(lbin[sum(range(alpha, alpha-i, -1)):(sum(range(alpha, alpha-i, -1))+alpha-i)], 2)
                Innprods = (Choices[i] @ np.fft.fftshift(W, axes=1).T) % 2
                h = np.zeros(alpha)
                if i >= len(Ibar):
                    h[Basis_free == Basis_free[i] - len(IbJb)] = 1
                Ch_i = Choices[i][np.all(np.mod(Innprods, 2) == h, axis=1), :]
                W[i, :] = Ch_i[vi_ind, :]

            # Fill the symplectic matrix
            Bl[np.ix_(np.concatenate([Ibar-1, m+Jbar-1]), )] = W
            F = (Ainv @ Bl) % 2
            F_all[l] = (F0 @ F) % 2

        return F_all

