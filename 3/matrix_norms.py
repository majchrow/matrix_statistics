import numpy as np
import numpy.linalg as la

THRESHOLD = 10 ** (-5)


def get_matrix(n):
    return np.random.rand(n, n)


def get_norm(xs, p):
    return np.sum(np.abs(xs) ** p, axis=0) ** (1 / p)


def get_infinity_norm(xs):
    return np.max(np.abs(xs), axis=0)


def get_normalized_vectors(n, p, samples):
    xs = np.random.rand(n, samples) * 2 -1
    norm_xs = get_norm(xs, p)
    normalized_xs = xs / norm_xs
    for i in range(samples):
        assert np.abs(la.norm(normalized_xs[:, i], p) - 1.0) < THRESHOLD
    return normalized_xs


def get_inf_normalized_vectors(n, samples):
    xs = np.random.rand(n, samples) * 2 -1
    norm_xs = get_infinity_norm(xs)
    normalized_xs = xs / norm_xs
    for i in range(samples):
        assert np.abs(la.norm(normalized_xs[:, i], np.inf) - 1.0) < THRESHOLD
    return normalized_xs


def get_approximate_matrix_p_norm(A, n, p, samples=100000):
    nxs = get_normalized_vectors(n, p, samples)
    Anxs = A @ nxs
    norm_Anxs = get_norm(Anxs, p)
    return np.max(norm_Anxs)


def get_approximate_matrix_infinity_norm(A, n, samples=100000):
    nxs = get_inf_normalized_vectors(n, samples)
    Anxs = A @ nxs
    norm_Anxs = get_infinity_norm(Anxs)
    return np.max(norm_Anxs)


if __name__ == "__main__":
    n = 3
    p = 2
    # A = get_matrix(n)
    A = np.array(
       [[4, 9, 2],
        [3, 5, 7],
        [8, 1, 6]]
    )
    print(f"A={A}")
    A_inv = la.inv(A)
    print(f"A^(-1)={A_inv}")

    for p in range(1, 3):
        p_norm = get_approximate_matrix_p_norm(A, n, p)
        p_norm_inv = get_approximate_matrix_p_norm(A_inv, n, p)

        print(f"||A||{p}={p_norm}")
        print(f"||A||{p}={la.norm(A, p)} (numpy)")
        print(f"||A^(-1)||{p}={p_norm_inv}")
        print(f"||A^(-1)||{p}={la.norm(A_inv, p)} (numpy)")

    inf_norm = get_approximate_matrix_infinity_norm(A, n)
    inf_norm_inv = get_approximate_matrix_infinity_norm(A_inv, n)
    print(f"||A||inf={inf_norm}")
    print(f"||A||inf={la.norm(A, np.inf)} (numpy)")
    print(f"||A^(-1)||inf={inf_norm_inv}")
    print(f"||A^(-1)||inf={la.norm(A_inv, np.inf)} (numpy)")
