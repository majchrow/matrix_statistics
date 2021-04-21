import numpy as np


def mm_block_svd(A, B, size, m, epsilon):
    assert size % m == 0
    blocks = int(size / m)
    C = np.zeros((size, size))

    UA_blocks, EA_blocks, VTA_blocks = _transform_to_svd_blocks(A, size, m, epsilon)
    UB_blocks, EB_blocks, VTB_blocks = _transform_to_svd_blocks(B, size, m, epsilon)

    for i in range(blocks):
        for j in range(blocks):
            C_block = _get_block(C, i, j, m)
            for k in range(blocks):
                UA, EA, VTA = UA_blocks[i][k], EA_blocks[i][k], VTA_blocks[i][k]
                UB, EB, VTB = UB_blocks[k][j], EB_blocks[k][j], VTB_blocks[k][j]

                C_block += _mb_svd(UA, EA, VTA, UB, EB, VTB)

    return C


def mm_block(A, B, size, m):
    assert size % m == 0
    blocks = int(size / m)
    C = np.zeros((size, size))

    A_blocks = _transform_to_blocks(A, size, m)
    B_blocks = _transform_to_blocks(B, size, m)

    for i in range(blocks):
        for j in range(blocks):
            C_block = _get_block(C, i, j, m)
            for k in range(blocks):
                C_block += _mb(A_blocks[i, k], B_blocks[k, j])

    return C


def _transform_to_blocks(M, size, m):
    blocks = int(size / m)
    M_blocks = np.zeros((blocks, blocks, m, m))

    for i in range(blocks):
        for j in range(blocks):
            M_blocks[i, j] = _get_block(M, i, j, m)
    return M_blocks


def _transform_to_svd_blocks(M, size, m, epsilon):
    blocks = int(size / m)
    U_matrices = [[] for _ in range(blocks)]
    E_matrices = [[] for _ in range(blocks)]
    VT_matrices = [[] for _ in range(blocks)]

    for i in range(blocks):
        for j in range(blocks):
            M_block = _get_block(M, i, j, m)
            U, E, VT = _svd(M_block, epsilon)
            U_matrices[i].append(U)
            E_matrices[i].append(E)
            VT_matrices[i].append(VT)

    return U_matrices, E_matrices, VT_matrices


def _mb(block1, block2):
    return block1 @ block2


def _get_block(M, i, j, block_size):
    return M[
        i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size
    ]


def _mb_svd(U1, E1, VT1, U2, E2, VT2):
    VT1 = E1 @ VT1
    VT2 = E2 @ VT2
    tmp = VT1 @ U2
    tmp = tmp @ VT2
    return U1 @ tmp


def _svd(A, epsilon):
    NU, tmp_e, NV_T = np.linalg.svd(A)
    index = len(tmp_e)

    for i, sv in enumerate(tmp_e):
        if np.abs(sv) < epsilon:
            index = i
            break

    NU = NU[:, :index]
    NV_T = NV_T[:index, :]
    NE = np.diag(tmp_e[:index])

    return NU, NE, NV_T


if __name__ == "__main__":
    A = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
    B = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    size = 4
    m = 2
    epsilon = 10**(-10)
    C1 = mm_block(A, B, size, m)
    C2 = mm_block_svd(A, B, size, m, epsilon)
    C3 = A @ B
    print(f"A={A}")
    print(f"B={B}")
    print(f"C={C1} (block)")
    print(f"C={C2} (block svd)")
    print(f"C={C3} (classic)")