import numpy as np
import scipy.linalg as la
from ristretto.ballistic.rblas.sketching import gaussian_operator
from ristretto.ballistic.randlapack.comps.rangefinders import FRRF,  \
    PowerRangeFinder
from ristretto.ballistic.randlapack.comps.powering import SORS, PoweredSketchOp


###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def qb(num_passes, A, k, rng):
    """
    Return matrices (Q, B) from a rank-k QB factorization of A.
    Use a Gaussian sketching matrix and pass over A a total of
    num_passes times.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A. We require num_passes >= 2.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    k : int
        Target rank for the approximation of A. Includes any oversampling.
        (E.g., if you want to be near the optimal (Eckhart-Young) error
        for a rank 20 approximation of A, then you might want to set k=25.)
        We require k <= min(A.shape).

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    Q : ndarray
        Has shape (A.shape[0], k). Columns are orthonormal.

    B : ndarray
        Has shape (k, A.shape[1]).

    Notes
    -----
    We perform (num_passes - 2) steps of subspace iteration, and
    stabilize subspace iteration by a QR factorization at every step.
    """
    rng = np.random.default_rng(rng)
    # Build the QB function
    rf = PowerRangeFinder(num_passes - 1, 1, orth, gaussian_operator)
    qb_ = FRQB(rf)
    # Call the QB function
    Q, B = qb_.exec(A, k, rng)
    return Q, B


def blk_qb_1(inner_num_pass, overwrite_A, A, blk, tol, max_rank, rng):
    """
    Iteratively build an approximate QB factorization of A,
    which terminates once either of the following conditions
    is satisfied
        (1)  || A - Q B ||_Fro <= tol
    or
        (2) Q has max_rank columns.

    Each iteration involves sketching A from the right by a sketching
    matrix with "blk" columns. The sketching matrix is constructed by
    applying (inner_num_pass - 2) steps of subspace iteration to a
    Gaussian matrix with blk columns.

    Parameters
    ----------
    inner_num_pass : int
        Number of passes over A in each iteration of this blocked QB
        algorithm. We require inner_num_pass >= 2.

    overwrite_A : bool
        If True, then this method modifies A in-place. If False, then
        we start the algorithm by constructing a complete copy of A.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    blk : int
        The block size in this blocked QB algorithm. Add this many columns
        to Q at each iteration (except possibly the final iteration).

    max_rank : int
        Terminate if Q.shape[1] == max_rank.

    tol : float
        Terminate if ||A - Q B||_Fro <= tol.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    Q : ndarray
        Has the same number of rows of A, and orthonormal columns.

    B : ndarray
        Has the same number of columns of A.

    Notes
    -----
    The number of columns in Q increase by "blk" at each iteration, unless
    that would bring Q.shape[1] > max_rank. In that case, the final
    iteration only adds enough columns to Q so that Q.shape[1] == max_rank.

    We perform (inner_num_pass - 2) steps of subspace iteration for each
    block of the QB factorization. We stabilize subspace iteration with
    QR factorization at each step.
    """
    rng = np.random.default_rng(rng)
    # Build the QB function
    rf = PowerRangeFinder(inner_num_pass, 1, orth, gaussian_operator)
    qb_ = BlockedQB1(rf, overwrite_A)
    # Call the QB function
    Q, B = qb_.exec(A, blk, tol, max_rank, rng)
    return Q, B


def blk_qb_2(num_passes, A, blk, tol, max_rank, rng):
    """
    Iteratively build an approximate QB factorization of A,
    which terminates once either of the following conditions
    is satisfied
        (1)  || A - Q B ||_Fro <= tol
    or
        (2) Q has max_rank columns.

    We start by obtaining a sketching matrix of shape
    (A.shape[1], max_rank), using (num_passes - 1) steps of
    subspace iteration on a random Gaussian matrix with
    max_rank columns. Then we perform two more passes over A
    before beginning iterative construction of (Q, B). Each
    iteration adds at most "blk" columns to Q and rows to B.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A in an efficient implementation
        of this algorithm (see Notes). We require num_passes >= 1.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    blk : int
        The block size in this blocked QB algorithm. Add this many columns
        to Q at each iteration (except possibly the final iteration).

    max_rank : int
        Terminate if Q.shape[1] == max_rank.

    tol : float
        Terminate if ||A - Q B||_Fro <= tol.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    Q : ndarray
        Has the same number of rows of A, and orthonormal columns.

    B : ndarray
        Has the same number of columns of A.

    Notes
    -----
    With its current implementation, this function requires num_passes + 1
    passes over A. An efficient implementation using two-in-one sketching
    could run this algorithm using only num_passes passes over A.

    We stabilize subspace iteration with a QR factorization at each step.
    """
    rng = np.random.default_rng(rng)
    sk_op = PoweredSketchOp(num_passes, 1, orth, gaussian_operator)
    Q, B = BlockedQB2(sk_op).exec(A, blk, tol, max_rank, rng)
    return Q, B


###############################################################################
#       Fixed rank QB
###############################################################################


class FRQB:

    def __init__(self, rf: FRRF):
        self.rangefinder = rf

    def exec(self, A, k, rng):
        """
        Return a rank-k approximation of A, represented by its
        factors in a QB decomposition. Construct the factor
        Q by calling this FRQB object's rangefinder.
        """
        rng = np.random.default_rng(rng)
        Q = self.rangefinder.exec(A, k, rng)
        B = Q.T @ A
        return Q, B


###############################################################################
#      Blocked QB: abstract base class, two implementations
###############################################################################

# TODO: maybe move "blk" into attributes of BlockedQB objects, rather than
#  the exec function? That would be in line with PoweredSketchOp, which has
#  tuning parameters (affecting speed and accuracy) in attributes rather than
#  exec. In that case the right name might be better as "adaptive QB" rather
#  than "blocked" QB.

class BaseBlockedQB:

    def exec(self, A, blk, tol, max_rank, rng):
        """
        Iteratively build an approximate QB factorization of A:
            The matrix Q has orthonormal columns.
            The matrix B is generally unstructured.
        Add at most "blk" columns to Q at each iteration.

        Stop once either of the following are satisfied
            (1)  || A - Q B || <= tol
        or
            (2) Q has max_rank columns.
        """
        raise NotImplementedError()


class BlockedQB1(BaseBlockedQB):

    def __init__(self, rf: FRRF, overwrite_a=False):
        self.rangefinder = rf
        self.overwrite_a = overwrite_a

    def exec(self, A, blk, tol, max_rank, rng):
        """
        Iteratively build an approximate QB factorization of A,
        which terminates once either of the following conditions
        is satisfied
            (1)  || A - Q B ||_Fro <= tol
        or
            (2) Q has max_rank columns.

        The algorithm starts by initializing empty matrices (Q, B).
        The i-th iteration of this algorithm uses this object's
        internal rangefinder to obtain a rank-"blk" QB factorization of A.
        Given the factors (Q_i, B_i) of the current iteration, we append
        Q_i to the columns to Q and we append B_i to the rows to B. Then we
        subtract Q_i @ B_i off from A and check termination criteria. If
        termination criteria aren't satisfied, we go to iteration (i+1).

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        blk : int
            The block size in this blocked QB algorithm. Add this many columns
            to Q at each iteration (except possibly the final iteration).

        max_rank : int
            Maximum number of columns in Q. Terminate if Q.shape[1] == max_rank.

        tol : float
            Terminate if ||A - Q B||_Fro <= tol.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.

        Returns
        -------
        Q : ndarray
            Has the same number of rows of A, and orthonormal columns.

        B : ndarray
            Has the same number of columns of A.

        Notes
        -----
        The number of columns in Q increase by "blk" at each iteration, unless
        that would bring Q.shape[1] > max_rank. In that case, the final
        iteration only adds enough columns to Q so that Q.shape[1] == max_rank.
        """
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
        rng = np.random.default_rng(rng)
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        sq_norm_A = la.norm(A, ord='fro') ** 2
        sq_tol = tol ** 2
        while True:
            if B.shape[0] + blk > max_rank:
                blk = int(max_rank - B.shape[0])  # final block
            # Standard QB, but step in to make extra sure that
            #   the columns of "Qi" are orthogonal to cols of current "Q".
            Qi = self.rangefinder.exec(A, blk, rng)
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

    def exec(self, A, blk, tol, max_rank, rng):
        """
        Iteratively build an approximate QB factorization of A,
        which terminates once either of the following conditions
        is satisfied
            (1)  || A - Q B ||_Fro <= tol
        or
            (2) Q has max_rank columns.

        We start by obtaining a sketching matrix of shape
        (A.shape[1], max_rank) from this object's sk_op.
        Then we perform two more passes over A before beginning
        iterative construction of (Q, B). Each iteration adds at
        most "blk" columns to Q and rows to B.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        blk : int
            The block size in this blocked QB algorithm. Add this many columns
            to Q at each iteration (except possibly the final iteration).

        max_rank : int
            Maximum number of columns in Q. Terminate if Q.shape[1] == max_rank.

        tol : float
            Terminate if ||A - Q B||_Fro <= tol.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.

        Returns
        -------
        Q : ndarray
            Has the same number of rows of A, and orthonormal columns.

        B : ndarray
            Has the same number of columns of A.

        Notes
        -----
        With its current implementation, this function requires at passes
        over A once the sketching matrix has been generated. An efficient
        implementation would only require one pass over A after the sketching
        matrix has been generated.
        """
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        if tol < np.inf:
            sq_norm_A = la.norm(A, ord='fro') ** 2
            sq_tol = tol**2
        rng = np.random.default_rng(rng)
        S = self.sk_op.exec(A, blk, rng)
        if not isinstance(S, np.ndarray):
            raise RuntimeError()
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
#      Helper functions
###############################################################################


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
