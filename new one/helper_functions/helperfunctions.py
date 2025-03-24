import numpy as np

class helper_functions:

    def __init__(self, m):
        """
        Initialize a SymplecticMatrix object with a given dimension 'm'.
        The symplectic matrix 'F' is initialized as the identity matrix.
        """
        self.m = m
        self.F = np.eye(2 * m, dtype=int)

    def mod2(matrix):
        """Perform element-wise mod 2 operation."""
        return np.mod(matrix, 2)

    def reduce_to_echelon_form(A):
        """
        Reduce the matrix A to echelon form over GF(2) and return the row operations matrix (M),
        column operations matrix (N), and the rank (rnk).
        """
        rows, cols = A.shape
        A = np.copy(A) % 2  # Ensure entries are in GF(2)

        # Initialize row and column operations as identity matrices
        M = np.eye(rows, dtype=int)  # Row operations matrix
        N = np.eye(cols, dtype=int)  # Column operations matrix

        row = 0  # Start from the top row

        for col in range(cols):
            if row >= rows:
                break  # Stop if all rows are processed

            # Find a pivot row for the current column
            pivot = np.where(A[row:, col] == 1)[0]
            if len(pivot) == 0:
                continue  # No pivot in this column, move to the next column

            pivot_row = pivot[0] + row  # Adjust pivot index to absolute position

            # Swap the current row with the pivot row
            A[[row, pivot_row]] = A[[pivot_row, row]]
            M[[row, pivot_row]] = M[[pivot_row, row]]  # Track row swap in M

            # Eliminate all other ones in this column
            for r in range(rows):
                if r != row and A[r, col] == 1:
                    A[r] = (A[r] + A[row]) % 2  # Row elimination using GF(2) addition
                    M[r] = (M[r] + M[row]) % 2  # Track the row operation in M

            # Track column swap in N if row != col (to ensure identity structure)
            if col != row:
                N[:, [col, row]] = N[:, [row, col]]  # Track column swap in N

            row += 1  # Move to the next row

        # Compute the rank as the number of non-zero rows in the reduced form
        rnk = np.sum(np.any(A, axis=1))

        return A, M, N, rnk

    def gf2matinv_v1(self, matrix):
        """
        Compute the inverse of a matrix in GF(2) using Gaussian elimination.
        """
        m = len(matrix)
        aug_matrix = np.concatenate((matrix, np.eye(m, dtype=int)), axis=1)
        for col in range(m):
            for row in range(col, m):
                if aug_matrix[row, col]:
                    break
        else:
        # If no row has a 1 in the current column, the matrix is singular in GF(2)
            raise ValueError("Matrix is singular over GF(2)")    
            
            if row != col:
                aug_matrix[[col, row]] = aug_matrix[[row, col]]
            for i in range(m):
                if i != col and aug_matrix[i, col]:
                    aug_matrix[i] ^= aug_matrix[col]
        return aug_matrix[:, m:]


    def gf2matin_v2(matrix):
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

    def apply_gate(self, gate, qubits):
        """
        Apply a quantum gate to the symplectic matrix 'F'.
        """
        I = np.eye(self.m, dtype=int)
        O = np.zeros((self.m, self.m), dtype=int)

        if gate.upper() == 'P':
            B = np.diag(np.mod(np.sum(I[qubits], axis=0), 2))
            FP = np.block([[I, B], [O, I]])
            self.F = np.mod(self.F @ FP, 2)

        elif gate.upper() == 'H':
            FH = np.eye(2 * self.m, dtype=int)
            indices = np.concatenate((qubits, self.m + qubits))
            FH[indices] = np.fft.fftshift(FH[indices], axes=1)
            self.F = np.mod(self.F @ FH, 2)

        elif gate.upper() == 'CZ':
            B = np.zeros((self.m, self.m), dtype=int)
            B[qubits[0], qubits[1]] = 1
            B[qubits[1], qubits[0]] = 1
            FCZ = np.block([[I, B], [O, I]])
            self.F = np.mod(self.F @ FCZ, 2)

        elif gate.upper() == 'CNOT':
            M = np.copy(I)
            M[qubits[0], qubits[1]] = 1
            FCNOT = np.block([[M, O], [O, self.gf2matinv(M).T]])
            self.F = np.mod(self.F @ FCNOT, 2)

        elif gate.upper() == 'PERMUTE':
            M = I[:, qubits]
            FPermute = np.block([[M, self.gf2matinv(M).T], [O, I]])
            self.F = np.mod(self.F @ FPermute, 2)

        else:
            print('\nfind_symplectic: Unrecognized gate encountered!\n')
            return None

    def find_symplectic(self, circuit):
        """
        Apply a sequence of quantum gates to obtain the final symplectic matrix.
        Return the symplectic matrix.
        """
        for gate, qubits in circuit:
            qubits = np.array(qubits) - 1
            self.apply_gate(gate, qubits)

        A = self.F[:self.m, :self.m]
        B = self.F[:self.m, self.m:]
        C = self.F[self.m:, :self.m]
        D = self.F[self.m:, self.m:]
        return self.F

# Example usage:
m_value = 1  # Replace this with your desired dimension
symplectic_object = SymplecticMatrix(m_value)
circuit_example = [('H', [1])]
result = symplectic_object.find_symplectic(circuit_example)
print(result)
