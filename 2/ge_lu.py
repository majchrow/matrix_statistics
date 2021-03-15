import numpy as np


def get_solution(A, b, size):
    x = np.zeros(size).astype('float64')
    for i in np.arange(size - 1, -1, -1):
        s = b[i]
        for j in np.arange(i + 1, size):
            s = s - A[i, j] * x[j]
        x[i] = s / A[i, i]
    return x


def ge_without_pivoting_and_with_ones_on_diagonal(A, b, size):
    for i in np.arange(size):
        # ones on diagonal
        tmp = A[i, i]
        A[i, i:] = A[i, i:] / tmp
        b[i] /= tmp
        # subtract i-th row (properly scaled to get zeros below diagonal) from consecutive rows
        for j in np.arange(i + 1, size):
            tmp = A[j, i]
            A[j, i:] = A[j, i:] - A[i, i:] * tmp
            b[j] -= b[i] * tmp
    print(f"{A}*x={b}")
    return get_solution(A, b, size)


def ge_without_pivoting_and_with_ones_on_diagonal_vectorized(A, b, size):
    for i in np.arange(size):
        # ones on diagonal
        tmp = A[i, i]
        A[i, i:] = A[i, i:] / tmp
        b[i] /= tmp
        # subtract i-th row (properly scaled to get zeros below diagonal) from consecutive rows
        # vectorized version
        tmp = A[i + 1:, i].copy()  # all the rows values to be subtracted
        A[i + 1:, i:] -= np.tile(A[i, i:], (tmp.shape[0], 1)) * tmp.reshape([-1, 1])
        b[i + 1:] -= b[i] * tmp
    print(f"{A}*x={b}")
    return get_solution(A, b, size)


def ge_without_pivoting(A, b, size):
    for i in np.arange(size - 1):
        # subtract i-th row (properly scaled to get zeros below diagonal) from consecutive rows
        for j in np.arange(i + 1, size):
            tmp = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - A[i, i:] * tmp
            b[j] -= b[i] * tmp
    print(f"{A}*x={b}")
    return get_solution(A, b, size)


def ge_without_pivoting_vectorized(A, b, size):
    for i in np.arange(size - 1):
        # subtract i-th row (properly scaled to get zeros below diagonal) from consecutive rows
        # vectorized version
        tmp = (A[i + 1:, i] / A[i, i]).copy()
        A[i + 1:, i:] -= np.tile(A[i, i:], (tmp.shape[0], 1)) * tmp.reshape([-1, 1])
        b[i + 1:] -= b[i] * tmp
    print(f"{A}*x={b}")
    return get_solution(A, b, size)


def ge_with_pivoting(A, b, size):
    for i in np.arange(size - 1):
        # find row with max value in given column
        pivot = np.argmax(A[i:, i]) + i
        print(f"Switching rows {i} and {pivot}")
        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]
        print(f"A after switching rows: {A}")
        print(f"b after switching rows: {b}")
        # subtract i-th row (properly scaled to get zeros below diagonal) from consecutive rows
        for j in np.arange(i + 1, size):
            tmp = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - A[i, i:] * tmp
            b[j] -= b[i] * tmp
    print(f"{A}*x={b}")
    return get_solution(A, b, size)


def ge_with_pivoting_vectorized(A, b, size):
    for i in np.arange(size - 1):
        # find row with max value in given column
        pivot = np.argmax(A[i:, i]) + i
        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]
        # subtract i-th row (properly scaled to get zeros below diagonal) from consecutive rows
        tmp = (A[i + 1:, i] / A[i, i]).copy()
        A[i + 1:, i:] -= np.tile(A[i, i:], (tmp.shape[0], 1)) * tmp.reshape([-1, 1])
        b[i + 1:] -= b[i] * tmp
    print(f"{A}*x={b}")
    return get_solution(A, b, size)


def LU_factorization(A):  # vectorized by default
    size = A.shape[0]
    L = np.eye(size)
    U = np.copy(A)
    for i in np.arange(size - 1):
        tmp = (U[i + 1:, i] / U[i, i]).copy()
        L[i + 1:, i] = tmp
        U[i + 1:, i:] -= np.tile(U[i, i:], (tmp.shape[0], 1)) * tmp.reshape([-1, 1])
    print(f"L={L}, U={U}")
    return L, U


if __name__ == "__main__":
    A = np.array([[1, 2, 4], [3, 5, 12], [20, 10, 1]]).astype('float64')
    b = np.array([1, 2, 7]).astype('float64')
    size = 3
    print("Gaussian elimination without pivoting (with ones on diagonal): ")
    solution1 = ge_without_pivoting_and_with_ones_on_diagonal_vectorized(A.copy(), b.copy(), size)
    print(f"x={solution1}")

    print("Gaussian elimination without pivoting: ")
    solution2 = ge_without_pivoting_vectorized(A.copy(), b.copy(), size)
    print(f"x={solution2}")

    print("Gaussian elimination with pivoting: ")
    solution3 = ge_with_pivoting_vectorized(A.copy(), b.copy(), size)
    print(f"x={solution3}")

    print("LU factorization without pivoting: ")
    L, U = LU_factorization(A)

    assert np.allclose(L @ U, A)

    print("Solution from numpy library: ")
    print(f"x={np.linalg.solve(A, b)}")

    # performance test
    # import time
    # C = np.random.randint(100000, size=(1000, 1000)).astype(np.float32)
    # d = np.random.randint(100000, size=(1000)).astype(np.float32)
    # start = time.time()
    # ge_without_pivoting_and_with_ones_on_diagonal_vectorized(C, d, C.shape[0])
    # print(f"Vectorized 1000x1000 time : {time.time() - start} s")
    # start = time.time()
    # ge_without_pivoting_and_with_ones_on_diagonal(C, d, C.shape[0])
    # print(f"Nonvectorized 1000x1000 time : {time.time() - start} s")
