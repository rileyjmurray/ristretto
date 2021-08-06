import numpy as np


def powered_range_sketch_op(A, k, num_pass, sketch_op_gen, stabilizer, pps, gen=None):
    """
    Use (num_pass - 1) passes over the matrix A to generate a matrix S where
    range(S) is (hopefully) closely aligned with the span of A's top right
    singular vectors. This is useful for estimating the span of the top
    *left* singular vectors of A, by evaluating Y = A @ S.

    We accomplish this roughly as follows:

        if num_pass is odd
            S = (A' A)^((num_pass - 1)//2) sketch_op_gen(n, k, generator)
        if num_pass is even
            S = (A' A)^((num_pass - 2)//2) A' sketch_op_gen(m, k, generator)

        where "generator := np.random.default_rng(gen)" is a Generator object
        from NumPy.random.

    That description is "rough" because repeated applications of A will cause
    floating point errors to rapidly accumulate. The parameter "pps" reads
    as "passes per stabilization": after "pps" applications of A or A.T, we
    call "stabilizer" on the working matrix to obtain a numerically well-
    behaved basis for its range. The most common choice of "stabilizer" is to
    return the factor Q from an (economic) QR factorization.
    """
    assert num_pass > 0
    gen = np.random.default_rng(gen)
    if num_pass % 2 == 1:
        S = sketch_op_gen(A.shape[1], k, gen)
        passes_done = 0
        q = (num_pass - 1) // 2
    else:
        S = A.T @ sketch_op_gen(A.shape[0], k, gen)
        passes_done = 1
        if pps == 1:
            S = stabilizer(S)
        q = (num_pass - 2) // 2
    # q is an even integer; need to compute
    #   S := (A' A)^q S
    # up to intermediate stabilization.
    while q > 0:
        S = A @ S
        passes_done += 1
        if passes_done % pps == 0:
            S = stabilizer(S)
        S = A.T @ S
        passes_done += 1
        if passes_done % pps == 0:
            S = stabilizer(S)
        q -= 1
    return S
