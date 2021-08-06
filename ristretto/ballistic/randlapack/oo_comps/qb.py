import numpy as np
import scipy.linalg as la
from ristretto.ballistic.rblas.sketching import gaussian_operator
from ristretto.ballistic.randlapack.oo_comps.rangefinders import FRRF, PowerRangeFinder
from ristretto.ballistic.randlapack.oo_comps.powering import SORS, PoweredSketchOp


def orth(S):
    return la.qr(S, mode='economic')


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


###############################################################################
#       Fixed rank QB
###############################################################################


class FRQB:

    def __init__(self, rf: FRRF):
        self.rf = rf

    def __call__(self, A, k, gen):
        """
        Return a rank-k approximation of A, represented by its
        factors in a QB decomposition.
        """
        gen = np.random.default_rng(gen)
        Q = self.rf(A, k, gen)
        B = Q.T @ A
        return Q, B


###############################################################################
#      Blocked QB: abstract base class, two implementations
###############################################################################

class BaseBlockedQB:

    def __call__(self, A, blk, tol, max_rank, gen):
        """
        Iteratively build an approximate QB factorization of A:
            The matrix Q has orthonormal columns.
            The matrix B is generally unstructured.
        Add "blk" columns to Q (resp. rows to B) at each iteration.

        Stop once either of the following are satisfied
            (1)  || A - Q B || <= tol
        or
            (2) Q has max_rank columns.
        """
        raise NotImplementedError()


class BlockedQB1(BaseBlockedQB):

    def __init__(self, rf: FRRF, overwrite_a=False):
        self.rf = rf
        self.overwrite_a = overwrite_a

    def __call__(self, A, blk, tol, max_rank, gen):
        if not self.overwrite_a:
            """
            TODO: Write a version of this function that doesn't
            need to copy A. Essentially, requires reaching inside
            power_rangefinder so that accesses to "A" are actually
            accesses to "A - Q @ B" for the current values of (Q, B).
            The Python implementation could do this pretty easily with
            a LinearOperator.
            """
            A = np.copy(A)
        gen = np.random.default_rng(gen)
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        sq_norm_A = la.norm(A, ord='fro') ** 2
        sq_tol = tol ** 2
        while True:
            if B.shape[0] + blk > max_rank:
                blk = int(max_rank - B.shape[0])  # final block
            # Standard QB, but step in to make extra sure that
            #   the columns of "Qi" are orthogonal to cols of current "Q".
            Qi = self.rf(A, blk, gen)
            Qi = project_out(Qi, Q, as_list=False)
            Qi = la.qr(Qi, mode='economic')[0]
            Bi = Qi.T @ A
            # Update the full factorization
            Q = np.column_stack((Q, Qi))
            B = np.row_stack((B, Bi))
            A = A - Qi @ Bi
            sq_norm_A = sq_norm_A - la.norm(Bi, ord='fro') ** 2
            if sq_norm_A <= sq_tol or B.shape[0] == max_rank:
                break
        return Q, B


class BlockedQB2(BaseBlockedQB):

    def __init__(self, sk_op: SORS):
        self.sk_op = sk_op

    def __call__(self, A, blk, tol, max_rank, gen):
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        if tol < np.inf:
            sq_norm_A = la.norm(A, ord='fro') ** 2
            sq_tol = tol**2
        gen = np.random.default_rng(gen)
        S = self.sk_op(A, blk, gen)
        G = A @ S
        H = A.T @ G
        for i in range(int(np.ceil(max_rank/blk))):
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
                sq_norm_A = sq_norm_A - la.norm(Bi, ord='fro')**2
                if sq_norm_A <= sq_tol:
                    break  # early stopping
        return Q, B


###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def qb(num_passes, A, k, gen):
    gen = np.random.default_rng(gen)
    # Build the QB function
    rf = PowerRangeFinder(num_passes, 1, orth, gaussian_operator)
    qb_obj = FRQB(rf)
    # Call the QB function
    Q, B = qb_obj(A, k, gen)
    return Q, B


def blk_qb_1(inner_num_pass, overwrite_A, A, blk, tol, max_rank, gen):
    gen = np.random.default_rng(gen)
    # Build the QB function
    rf = PowerRangeFinder(inner_num_pass, 1, orth, gaussian_operator)
    qb = BlockedQB1(rf, overwrite_A)
    # Call the QB function
    Q, B = qb(A, blk, tol, max_rank, gen)
    return Q, B


def blk_qb_2(num_passes, A, blk, tol, max_rank, gen):
    gen = np.random.default_rng(gen)
    # Build the QB function
    sk_op = PoweredSketchOp(num_passes, 1, orth, gaussian_operator)
    qb = BlockedQB2(sk_op)
    # Call the QB function
    Q, B = qb(A, blk, tol, max_rank, gen)
    return Q, B
