import numpy as np
from typing import List, Tuple, Union
from helper_functions.helperfunctions import gf2_matinv, gf2_rref, gf2lu

class SymplecticCircuitBuilder:
    def __init__(self):
        pass

    def symp_mat_decompose(self, F: np.ndarray) -> List[np.ndarray]:
        m = F.shape[0] // 2
        I = np.eye(m, dtype=int)
        O = np.zeros((m, m), dtype=int)
        A = F[:m, :m]
        B = F[:m, m:]
        C = F[m:, :m]
        D = F[m:, m:]

        if (np.array_equal(A, I) and np.array_equal(C, O) and np.array_equal(D, I)) or \
           (np.array_equal(B, O) and np.array_equal(C, O)) or \
           (np.array_equal(F, np.block([[O, I], [I, O]]))):
            return [F]

        Omega = np.block([[O, I], [I, O]])

        def Elem1(Q):
            return np.block([[Q, np.zeros_like(Q)],
                             [np.zeros_like(Q), gf2_matinv(Q.T)]])

        def Elem2(R):
            return np.block([[I, R], [O, I]])

        def U(k): return np.vstack([
            np.hstack([np.eye(k, dtype=int), np.zeros((k, m-k), dtype=int)]),
            np.zeros((m-k, m), dtype=int)
        ])

        def L(k): return np.vstack([
            np.zeros((m-k, m), dtype=int),
            np.hstack([np.zeros((k, m-k), dtype=int), np.eye(k, dtype=int)])
        ])

        def Elem3(k):
            return np.block([[L(m-k), U(k)], [U(k), L(m-k)]])

        _, M_A, N_A, k = gf2_rref(A)
        Qleft1 = Elem1(M_A)
        Qright = Elem1(N_A)
        Fcp = Qleft1 @ F @ Qright % 2

        if k == m:
            Rright = Elem2(Fcp[:m, m:])
            Fcp = Fcp @ Rright % 2
            R = Fcp[m:, :m]
            return [gf2_matinv(Qleft1), Omega, Elem2(R), Omega, gf2_matinv(Rright), gf2_matinv(Qright)]

        Bmk = Fcp[k:m, m+k:m]
        _, M_Bmk1, _, _ = gf2_rref(Bmk)
        M_Bmk = np.block([
            [np.eye(k, dtype=int), np.zeros((k, m-k), dtype=int)],
            [np.zeros((m-k, k), dtype=int), M_Bmk1]
        ])
        Qleft2 = Elem1(M_Bmk)
        Fcp = Qleft2 @ Fcp % 2

        E = Fcp[:k, m+k:]
        M_E = np.block([
            [np.eye(k, dtype=int), E],
            [np.zeros((m-k, k), dtype=int), np.eye(m-k, dtype=int)]
        ])
        Qleft3 = Elem1(M_E)
        Fcp = Qleft3 @ Fcp % 2

        S = Fcp[:k, m:m+k]
        upper_block = np.hstack([S, np.zeros((k, m-k), dtype=int)])
        lower_block = np.zeros((m-k, m), dtype=int)
        Rright = Elem2(np.vstack([upper_block, lower_block]))
        Fcp = Fcp @ Rright % 2

        Fright = Omega @ Elem3(k) % 2
        Fcp = Fcp @ Fright % 2
        R = Fcp[m:, :m]

        Q = Qleft3 @ Qleft2 @ Qleft1 % 2
        return [gf2_matinv(Q), Omega, Elem2(R), Elem3(k), gf2_matinv(Rright), gf2_matinv(Qright)]

    def find_circuit(self, F: np.ndarray) -> List[Tuple[str, Union[List[int], Tuple[int, int]]]]:
        m = F.shape[0] // 2
        I = np.eye(m, dtype=int)
        O = np.zeros((m, m), dtype=int)
        Omega = np.block([[O, I], [I, O]])

        if not np.array_equal(F @ Omega @ F.T % 2, Omega):
            print("\nInvalid symplectic matrix!")
            return []

        if np.array_equal(F, np.eye(2*m, dtype=int)):
            return []

        Decomp = self.symp_mat_decompose(F)
        circuit = []

        for mat in Decomp:
            if np.array_equal(mat, np.eye(2*m, dtype=int)):
                continue
            elif np.array_equal(mat, Omega):
                circuit.append(('H', list(range(1, m+1))))
                continue

            A = mat[:m, :m]
            B = mat[:m, m:]
            C = mat[m:, :m]
            D = mat[m:, m:]

            if np.array_equal(A, I) and np.array_equal(C, O) and np.array_equal(D, I):
                P_ind = [i+1 for i in range(m) if B[i, i] == 1]
                if P_ind:
                    circuit.append(('P', P_ind))

                B_upper = np.triu((B + np.diag(np.diag(B))) % 2)
                for j in range(m):
                    CZ_ind = np.where(B_upper[j] == 1)[0]
                    for k in CZ_ind:
                        circuit.append(('CZ', (j+1, k+1)))

            elif np.array_equal(B, O) and np.array_equal(C, O):
                L, U, P = gf2lu(A)
                if not np.array_equal(P, I):
                    perm = list((np.arange(m) @ P.T) + 1)
                    circuit.append(('Permute', perm))

                for j in range(m):
                    inds = [i for i in np.where(L[j] == 1)[0] if i != j]
                    for k in inds:
                        circuit.append(('CNOT', (j+1, k+1)))

                for j in reversed(range(m)):
                    inds = [i for i in np.where(U[j] == 1)[0] if i != j]
                    for k in inds:
                        circuit.append(('CNOT', (j+1, k+1)))

            else:
                k = m - np.trace(A)
                Uk = np.vstack([np.hstack([np.eye(k, dtype=int), np.zeros((k, m-k), dtype=int)]),
                                np.zeros((m-k, m), dtype=int)])
                Lmk = np.vstack([np.zeros((k, m), dtype=int),
                                 np.hstack([np.zeros((m-k, k), dtype=int), np.eye(m-k, dtype=int)])])
                if np.array_equal(A, Lmk) and np.array_equal(B, Uk) and \
                   np.array_equal(C, Uk) and np.array_equal(D, Lmk):
                    circuit.append(('H', list(range(1, k+1))))
                else:
                    print("\nUnknown elementary symplectic form!")
                    return []

        return circuit
