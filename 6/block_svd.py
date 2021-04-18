import numpy as np

def mm_block(A,B,size,m):
    assert size % m == 0
    blocks = int(size/m)
    C = np.zeros((size,size))

    for i in range(blocks):
        for j in range(blocks):
            C_block = _get_block(C,i,j,m)
            for k in range(blocks):
                A_block = _get_block(A,i,k,m)
                B_block = _get_block(B,k,j,m)
                C_block += _mb(A_block,B_block)

    return C

def mm_block_svd(A,B,size,m,epsilon):
    assert size % m == 0
    blocks = int(size/m)
    C = np.zeros((size,size))

    for i in range(blocks):
        for j in range(blocks):
            C_block = _get_block(C,i,j,m)
            for k in range(blocks):
                A_block = _get_block(A,i,k,m)
                UA, EA, VTA = _svd(A_block,epsilon)

                B_block = _get_block(B,k,j,m)
                UB, EB, VTB = _svd(B_block,epsilon)

                C_block += _mb_svd(UA,EA,VTA,UB,EB,VTB)

    return C

def _mb(block1,block2):
    return block1 @ block2

def _get_block(M,i,j,block_size):
    return M[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]

def _mb_svd(U1,E1,VT1,U2,E2,VT2):
    VT1 = E1 @ VT1
    VT2 = E2 @ VT2
    tmp = VT1 @ U2
    tmp = tmp @ VT2
    return U1 @ tmp

def _svd(A, epsilon):
    NU, tmp_e, NV_T = np.linalg.svd(A)
    
    for i,sv in enumerate(tmp_e):
        if np.abs(sv) < epsilon:
            break

    NU = NU[:,:i]
    NV_T = NV_T[:i,:]
    NE = np.diag(tmp_e[:i])
    
    return NU, NE, NV_T

if __name__ == "__main__":
    A = np.array([[0,0,0,1],[0,1,0,0],[0,0,1,0],[1,0,0,0]])
    B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    size = 4
    m = 2
    epsilon = 10**(-20)
    C1 = mm_block(A,B,size,m)
    C2 = mm_block_svd(A,B,size,m,epsilon)
    C3 = A @ B
    print(f"A={A}")
    print(f"B={B}")
    print(f"C={C1} (block)")
    print(f"C={C2} (block svd)")
    print(f"C={C3} (classic)")