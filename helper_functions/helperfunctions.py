import numpy as np

class SymplecticMatrix:
    def __init__(self, m):
        """
        Initialize a SymplecticMatrix object with a given dimension 'm'.
        The symplectic matrix 'F' is initialized as the identity matrix.
        """
        self.m = m
        self.F = np.eye(2 * m, dtype=int)

    def gf2matinv(self, matrix):
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
