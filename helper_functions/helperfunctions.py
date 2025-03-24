import numpy as np

def gf2_matinv(A):
    """Compute the inverse of a binary matrix A over GF(2)."""
    
    A = np.array(A, dtype=np.uint8) % 2  # Ensure elements are 0 or 1 in GF(2)
    n = A.shape[0]
    I = np.eye(n, dtype=int)
    AI = np.concatenate((A.copy(), I), axis=1)
    for i in range(n):
        if AI[i, i] == 0:
            for j in range(i + 1, n):
                if AI[j, i] == 1:
                    AI[[i, j]] = AI[[j, i]]
                    break
        for j in range(n):
            if i != j and AI[j, i] == 1:
                AI[j] ^= AI[i]
    return AI[:, n:]

def gf2_rref(A):
    """Compute the RREF of binary matrix A over GF(2)."""
    A = np.array(A, dtype=np.uint8) % 2  # Ensure elements are 0 or 1 in GF(2)
    A = A.copy()
    m, n = A.shape
    i = j = 0
    M = np.eye(m, dtype=int)
    N = np.eye(n, dtype=int)
    while i < m and j < n:
        if A[i, j] == 0:
            for k in range(i+1, m):
                if A[k, j] == 1:
                    A[[i, k]] = A[[k, i]]
                    M[[i, k]] = M[[k, i]]
                    break
        if A[i, j] == 1:
            for k in range(m):
                if k != i and A[k, j] == 1:
                    A[k] ^= A[i]
                    M[k] ^= M[i]
            for k in range(n):
                if k != j and A[i, k] == 1:
                    A[:, k] ^= A[:, j]
                    N[:, k] ^= N[:, j]
            i += 1
        j += 1
    rank = i
    return A, M, N, rank

def gf2lu(A):
    """
    Perform LU decomposition of a binary matrix A over GF(2).
    Returns L, U, P such that P @ A = L @ U.
    L is lower-triangular, U is upper-triangular, and P is a permutation matrix.
    """
    m = A.shape[0]
    U = A.copy()
    L = np.eye(m, dtype=int)
    P = np.eye(m, dtype=int)

    for k in range(m - 1):
        pivot = np.where(U[k:, k] == 1)[0]
        if pivot.size == 0:
            continue
        i = pivot[0] + k
        U[[k, i], k:] = U[[i, k], k:]
        L[[k, i], :k] = L[[i, k], :k]
        P[[k, i]] = P[[i, k]]
        for j in range(k + 1, m):
            L[j, k] = U[j, k]
            if L[j, k] == 1:
                U[j, k:] ^= U[k, k:]
    return L, U, P

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
                A[r] = (A[r] + A[row]) % 2
                b[r] = (b[r] + b[row]) % 2
        row += 1
    return A, b

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

def gf2_gaussian_elimination_with_echelon(A, b):
    A_echelon, b_echelon = reduce_to_echelon_form(A, b)
    return solve_from_echelon_form(A_echelon, b_echelon), True
