import numpy as np
import time

def _get_shapes(m_a, m_b):  # SHAPE A:(NxM) B:(MxL) C: NxL
    N = len(m_a)
    M = len(m_b)
    L = len(m_b[0])
    return N, M, L


def _init_zeros(N, L):
    return [[0. for _ in range(L)] for _ in range(N)]


def mm_ijk(m_a, m_b):
    N, M, L = _get_shapes(m_a, m_b)

    m_c = _init_zeros(N, L)

    for i in range(N):  # rows
        for j in range(L):  # columns
            for k in range(M):  # dot product
                m_c[i][j] += m_a[i][k] * m_b[k][j]
    return m_c


def mm_ikj(m_a, m_b):
    N, M, L = _get_shapes(m_a, m_b)

    m_c = _init_zeros(N, L)

    for i in range(N):  # rows
        for k in range(M):  # dot product
            for j in range(L):  # columns
                m_c[i][j] += m_a[i][k] * m_b[k][j]

    return m_c


def mm_jik(m_a, m_b):
    N, M, L = _get_shapes(m_a, m_b)

    m_c = _init_zeros(N, L)

    for j in range(L):  # columns
        for i in range(N):  # rows
            for k in range(M):  # dot product
                m_c[i][j] += m_a[i][k] * m_b[k][j]

    return m_c


def mm_jki(m_a, m_b):
    N, M, L = _get_shapes(m_a, m_b)

    m_c = _init_zeros(N, L)

    for j in range(L):  # columns
        for k in range(M):  # dot product
            for i in range(N):  # rows
                m_c[i][j] += m_a[i][k] * m_b[k][j]

    return m_c


def mm_kij(m_a, m_b):
    N, M, L = _get_shapes(m_a, m_b)

    m_c = _init_zeros(N, L)

    for k in range(M):  # dot product
        for i in range(N):  # rows
            for j in range(L):  # columns
                m_c[i][j] += m_a[i][k] * m_b[k][j]

    return m_c


def mm_kji(m_a, m_b):
    N, M, L = _get_shapes(m_a, m_b)

    m_c = _init_zeros(N, L)

    for k in range(M):  # dot product
        for j in range(L):  # columns
            for i in range(N):  # rows
                m_c[i][j] += m_a[i][k] * m_b[k][j]

    return m_c


def multiply_matrices_and_measure_time(mm, m_a, m_b):
    current_time = lambda: time.time() * 1000
    start_time = current_time()
    m_c = mm(m_a, m_b)
    end_time = current_time()
    return m_c, round(end_time - start_time,3)

def generate_matrix(size):
    return np.random.rand(size,size)

def run_tests(size):
    print(f"RUNNING TESTS FOR MATRIX SIZE: {size}")
    m_a = generate_matrix(size)
    m_b = generate_matrix(size)

    mm_operations = [("IJK", mm_ijk), ("IKJ", mm_ikj), ("JIK", mm_jik), ("JKI", mm_jki), ("KIJ", mm_kij), ("KJI", mm_kji)]
    resutlt_matrices = []

    for loops_sequence, mm in mm_operations:
        m_c, op_time = multiply_matrices_and_measure_time(mm, m_a, m_b)
        print(f"{loops_sequence} - TIME: {op_time}")
        resutlt_matrices.append(m_c)

    base_result = resutlt_matrices[0]

    for result_matrix in resutlt_matrices[1:]:
        assert np.allclose(base_result, result_matrix)

def main():
    m_a = [
        [1, 1, 1],
        [3, 1, 1],
        [4, 3, 1],
        [1, 5, 1]
    ]

    m_b = [
        [2, 1],
        [3, 2],
        [4, 1]
    ]

    # ijk
    m_c = mm_ijk(m_a, m_b)
    print("IJK:", np.matrix(m_c), sep='\n')

    # ikj
    m_c = mm_ikj(m_a, m_b)
    print("IKJ:", np.matrix(m_c), sep='\n')

    # jik
    m_c = mm_jik(m_a, m_b)
    print("JIK:", np.matrix(m_c), sep='\n')

    # jki
    m_c = mm_jki(m_a, m_b)
    print("JKI:", np.matrix(m_c), sep='\n')

    # kij
    m_c = mm_kij(m_a, m_b)
    print("KIJ:", np.matrix(m_c), sep='\n')

    # kji
    m_c = mm_kji(m_a, m_b)
    print("KJI:", np.matrix(m_c), sep='\n')


if __name__ == '__main__':
    sizes = [10,100,1000]
    for size in sizes:
        run_tests(size)
