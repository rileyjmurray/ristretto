import numpy as np
import scipy.linalg as la


def powered_range_sketch_op(A, k, num_pass, sketch_op_gen, stabilizer, pps):
    """
    Use (num_pass - 1) passes over the matrix A to generate a matrix S where
    range(S) is (hopefully) closely aligned with the span of A's top right
    singular vectors. This is useful for estimating the span of the top
    *left* singular vectors of A, by evaluating Y = A @ S.

    We accomplish this roughly as follows:

        if num_pass is odd
            S = (A' A)^((num_pass - 1)//2) sketch_op_gen(n, k)
        if num_pass is even
            S = (A' A)^((num_pass - 2)//2) A' sketch_op_gen(m, k)

    That description is "rough" because repeated applications of A will cause
    floating point errors to rapidly accumulate. The parameter "pps" reads
    as "passes per stabilization": after "pps" applications of A or A.T, we
    call "stabilizer" on the working matrix to obtain a numerically well-
    behaved basis for its range. The most common choice of "stabilizer" is to
    return the factor Q from an (economic) QR factorization.
    """
    assert num_pass > 0
    if num_pass % 2 == 1:
        S = sketch_op_gen(A.shape[1], k)
        passes_done = 0
        q = (num_pass - 1) // 2
    else:
        S = A.T @ sketch_op_gen(A.shape[0], k)
        passes_done = 1
        if pps == 1:
            S = stabilizer(S)
        q = (num_pass - 2) // 2
    # q is an even integer; need to compute
    #   S := (A' A)^q S and
    # up to intermediate stabilization.
    if q > 0:
        S = charge_sample(A, S, q, stabilizer, pps, passes_done)
    return S


def charge_sample(A, S, q, stabilizer, pps, pass_count):
    """
    Construct a matrix T where range(T) = range( (A' A)^q S ).
    Call the stabilizer every "pps" passes over the matrix A,
    assuming "pass_count" number of passes have occurred in the
    calling function.
    """
    while q > 0:
        S = A @ S
        pass_count += 1
        if pass_count % pps == 0:
            S = stabilizer(S)
        S = A.T @ S
        pass_count += 1
        if pass_count % pps == 0:
            S = stabilizer(S)
        q -= 1
    return S
