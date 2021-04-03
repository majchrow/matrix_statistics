import numpy as np


def SVD(A, verbose=False):
    # Calculate eigenvalues and eigenvectors for AA^T and A^TA
    if verbose:
        print("Power method for AA^T")
    eigenvalues, eigenvectors_AAt = power_method(A @ A.T, verbose=verbose)
    if verbose:
        print("Power method for A^TA")
    _, eigenvectors_AtA = power_method(A.T @ A, verbose=verbose)

    # U matrix
    U = np.column_stack(eigenvectors_AAt)

    # Sigma matrix
    E = np.diag(np.sqrt(eigenvalues))

    # V matrix
    V = np.column_stack(eigenvectors_AtA)

    return U, E, V


def power_method(A, verbose=False):
    eigenvalues = []
    eigenvectors = []
    E = A.copy()

    for i in range(A.shape[0]):
        eigenvalue, eigenvector = power_iteration(E, verbose=verbose)

        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        if i < A.shape[0]:
            E -= eigenvalue * np.outer(eigenvector, eigenvector)

    return eigenvalues, eigenvectors


def power_iteration(A, max_error=1e-12, max_iter=500, start=None, verbose=False):
    if start is None:
        z = np.random.rand(A.shape[0])
    else:
        z = start

    for iter in range(max_iter):
        w = A @ z
        lamb = max(np.abs(w))
        err = np.linalg.norm(w - lamb * z)
        if err < max_error:
            if verbose:
                print(f"Found eigenvalue {lamb} with error {err} after {iter + 1} iterations")
            break
        z = w / lamb
    else:
        if verbose:
            print(f"Eigenvalue not found with error less than {err} after {max_iter} iterations")

    eigenvalue = lamb
    eigenvector = z / np.linalg.norm(z)
    return eigenvalue, eigenvector


def main(verbose=False, test_power=False, test_svd=False, print_numpy=False):
    A = np.array(
        [[2, 22, 11],
         [-10, 21, 10],
         [-11, -1, -2]]
    ).astype(float)

    # A = np.array(
    #     [[3, 1],
    #      [6, 2]]
    # ).astype(float)

    # A = np.array(
    #     [[2, -1, 0],
    #      [-1, 2, -1],
    #      [0, -1, 2]]
    # ).astype(float)

    if test_power:
        # Power method test
        print("==================================")
        print("Testing power method for matrix A")
        eig_numpy, _ = np.linalg.eig(A)
        eig_ours, _ = power_method(A, verbose=verbose)
        print("Numpy eigenvalues:", eig_numpy)
        print("Ours  eigenvalues:", eig_ours)
        print("Done")

    print("==================================")
    print("Testing SVD")

    U, E, V = SVD(A, verbose=verbose)
    UEVT = U @ E @ V.T
    print("U:\n", np.around(U, 2))
    print("SIGMA:\n", np.around(E, 2))  # Expected diagonal
    print("V:\n", np.around(V, 2))
    print("A:\n", np.around(A, 2))
    print("UEV^t:\n", np.around(UEVT, 2))  # Expected A
    print("VV^t:\n", np.around(V @ V.T, 2))  # Expected identity (with nonzero eigenvalues)
    print("UU^t:\n", np.around(U @ U.T, 2))  # Expected identity (with nonzero eigenvalues)

    print("Done")

    NU, tmp_e, NV_T = np.linalg.svd(A)
    NE = np.diag(tmp_e)
    NUEVT = NU @ NE @ NV_T

    if print_numpy:
        print("==================================")
        print("Testing Numpy SVD")
        print("Numpy U:\n", np.around(NU, 2))
        print("Numpy SIGMA:\n", np.around(NE, 2))
        print("Numpy V^T:\n", np.around(NV_T, 2))
        print("A:\n", np.around(A, 2))
        print("Numpy UEV^t\n", np.around(NUEVT))

    if test_svd:
        print("abs(U) - abs(U(NUMPY)):", np.around(np.sum(np.abs(np.abs(U) - np.abs(NU))), 2))
        print("abs(SIGMA) - abs(SIGMA(NUMPY)):", np.around(np.sum(np.abs(np.abs(E) - np.abs(NE))), 2))
        print("abs(V^t) - abs(V^t(NUMPY)):", np.around(np.sum(np.abs(np.abs(V.T) - np.abs(NV_T))), 2))
        print("abs(UEV^t) - abs(UEV^t(NUMPY)):", np.around(np.sum(np.abs(np.abs(UEVT) - np.abs(NUEVT))), 2))


if __name__ == '__main__':
    main(verbose=True, test_power=True, test_svd=True, print_numpy=True)
