import numpy as np
import scipy.linalg as la
from ristretto.ballistic.randlapack.comps.powering import powered_range_sketch_op
from ristretto.ballistic.rblas import sketching as sk
from ristretto.ballistic.randlapack.comps.rangefinders import power_rangefinder


def default_sketch_stab(sketch_op, stabilizer):
    if stabilizer is None:
        def stabilizer(mat):
            return la.qr(mat, mode='economic')[0]
    if sketch_op is None:
        sketch_op = sk.gaussian_operator
    return sketch_op, stabilizer


def qb(A, k, num_pass, sketch_op=None, gen=None, stabilizer=None, pps=1):
    sketch_op, stabilizer = default_sketch_stab(sketch_op, stabilizer)
    gen = np.random.default_rng(gen)
    Q = power_rangefinder(A, k, num_pass, sketch_op, gen, pps)
    B = Q.T @ A
    return Q, B


def qb_b_fet(A, blk, tol, p,
             overwrite_a=False, sketch_op=None, gen=None,
             stabilizer=None, pps=1, max_rank=np.inf):
    if not overwrite_a:
        #TODO: Write a version of this function that doesn't
        # need to copy A. Essentially, requires reaching inside
        # power_rangefinder so that accesses to "A" are actually
        # accesses to "A - Q @ B" for the current values of (Q, B).
        # The Python implementation could do this pretty easily with
        # a LinearOperator.
        A = np.copy(A)
    sketch_op, stabilizer = default_sketch_stab(sketch_op, stabilizer)
    gen = np.random.default_rng(gen)
    Q = np.empty(shape=(A.shape[0], 0), dtype=float)
    B = np.empty(shape=(0, A.shape[1]), dtype=float)
    sqnorm_A = la.norm(A, ord='fro') ** 2
    sqtol = tol**2
    while True:
        if B.shape[0] + blk > max_rank:
            blk = int(max_rank - B.shape[0])  # final block
        # Standard QB, but step in to make extra sure that
        #   the columns of "Qi" are orthogonal to cols of current "Q".
        Qi = power_rangefinder(A, blk, p, sketch_op, stabilizer, pps, gen)
        Qi = project_out(Qi, Q, as_list=False)
        Qi = la.qr(Qi, mode='economic')[0]
        Bi = Qi.T @ A
        # Update the full factorization
        Q = np.column_stack((Q, Qi))
        B = np.row_stack((B, Bi))
        A = A - Qi @ Bi
        sqnorm_A = sqnorm_A - la.norm(Bi, ord='fro')**2
        if sqnorm_A <= sqtol or B.shape[0] == max_rank:
            break
    return Q, B


def qb_b_pe(A, blk, ell, p, sketch_op=None, gen=None, stabilizer=None, pps=1, tol=np.inf):
    sketch_op, stabilizer = default_sketch_stab(sketch_op, stabilizer)
    Q = np.empty(shape=(A.shape[0], 0), dtype=float)
    B = np.empty(shape=(0, A.shape[1]), dtype=float)
    if tol < np.inf:
        sqnorm_A = la.norm(A, ord='fro') ** 2
        sqtol = tol**2
    gen = np.random.default_rng(gen)
    S = powered_range_sketch_op(A, ell, p, sketch_op, gen, stabilizer, pps)
    G = A @ S
    H = A.T @ G
    for i in range(int(np.ceil(ell/blk))):
        blk_start = i*blk
        blk_end = min((i+1)*blk, S.shape[1])
        Si = S[:, blk_start:blk_end]
        BSi = B @ Si
        Yi = G[:, blk_start:blk_end] - Q @ BSi
        Qi, Ri = la.qr(Yi, mode='economic')
        Qi = project_out(Qi, Q, as_list=False)  # Qi = Qi - Q @ (Q.T @ Qi)
        Qi, Rihat = la.qr(Qi, mode='economic')
        Ri = Rihat @ Ri
        Bi = H[:, blk_start:blk_end].T - (Yi.T @ Q) @ B - BSi.T @ B
        la.solve_triangular(Ri, Bi, trans='T', overwrite_b=True)
        Q = np.column_stack((Q, Qi))
        B = np.row_stack((B, Bi))
        if tol < np.inf:
            sqnorm_A = sqnorm_A - la.norm(Bi, ord='fro')**2
            if sqnorm_A <= sqtol:
                break  # early stopping
    return Q, B


def project_out(Qi, Q, as_list=False):
    #TODO: perform operation in-place.
    if as_list:
        #TODO: implement and use in qb_b_fet.
        # NOTE: Q is accessed in a few different places in
        #       qb_b_pe, so this wouldn't be enough to avoid
        #       updating Q to be contiguous at each iteration.
        raise NotImplementedError()
    else:
        Qi = Qi - Q @ (Q.T @ Qi)
        return Qi
