import numpy as np


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
    main()
