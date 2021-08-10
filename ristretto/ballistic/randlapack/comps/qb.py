import numpy as np
import scipy.linalg as la
from ristretto.ballistic.rblas.sketching import gaussian_operator
from ristretto.ballistic.randlapack.comps.rangefinders import RangeFinder,  \
    RF1
from ristretto.ballistic.randlapack.comps.powering import SORS, PoweredSketchOp
import ristretto.ballistic.randlapack.utilities as util


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
    rf = RF1(num_passes - 1, 1, util.orth, gaussian_operator)
    Q, B = QB1(rf).exec(A, k, np.inf, True, rng)
    return Q, B


def qb_b_fet(inner_num_pass, blk, overwrite_A, A, k, tol, rng):
    """
    Iteratively build an approximate QB factorization of A,
    which terminates once either of the following conditions
    is satisfied
        (1)  || A - Q B ||_Fro <= tol
    or
        (2) Q has k columns.

    Each iteration involves sketching A from the right by a sketching
    matrix with "blk" columns. The sketching matrix is constructed by
    applying (inner_num_pass - 2) steps of subspace iteration to a
    Gaussian matrix with blk columns.

    Parameters
    ----------
    inner_num_pass : int
        Number of passes over A in each iteration of this blocked QB
        algorithm. We require inner_num_pass >= 2.

    blk : int
        The block size in this blocked QB algorithm. Add this many columns
        to Q at each iteration (except possibly the final iteration).

    overwrite_A : bool
        If True, then this method modifies A in-place. If False, then
        we start the algorithm by constructing a complete copy of A.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    k : int
        Terminate if Q.shape[1] == k.

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
    that would bring Q.shape[1] > k. In that case, the final iteration only
    adds enough columns to Q so that Q.shape[1] == k.

    We perform (inner_num_pass - 2) steps of subspace iteration for each
    block of the QB factorization. We stabilize subspace iteration with
    QR factorization at each step.
    """
    rng = np.random.default_rng(rng)
    rf = RF1(inner_num_pass, 1, util.orth, gaussian_operator)
    Q, B = QB2(rf, blk, overwrite_A).exec(A, k, tol, True, rng)
    return Q, B


def qb_b_pe(num_passes, blk, A, k, tol, rng):
    """
    Iteratively build an approximate QB factorization of A,
    which terminates once either of the following conditions
    is satisfied
        (1)  || A - Q B ||_Fro <= tol
    or
        (2) Q has k columns.

    We start by obtaining a sketching matrix of shape (A.shape[1], k),
    using (num_passes - 1) steps of subspace iteration on a random Gaussian
    matrix with k columns. Then we perform two more passes over A before
    beginning iterative construction of (Q, B). Each iteration adds at most
    "blk" columns to Q and rows to B.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A in an efficient implementation
        of this algorithm (see Notes). We require num_passes >= 1.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    blk : int
        The block size in this blocked QB algorithm. Add this many
        columns to Q at each iteration (except possibly the final iteration).

    k : int
        Terminate if Q.shape[1] == k.

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
    sk_op = PoweredSketchOp(num_passes, 1, util.orth, gaussian_operator)
    Q, B = QB3(sk_op, blk).exec(A, k, tol, True, rng)
    return Q, B


###############################################################################
#       Object-oriented interfaces
###############################################################################

class QBFactorizer:

    def exec(self, A, k, tol, eager, rng):
        """
        Return a matrix Q with orthonormal columns and a matrix B where
        the product Q B stands in as an approximation of A.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to be approximated.

        k : int
            Target for the number of columns in Q. Must be <= A.shape[0].

        tol : float
            Target for the error || A - Q B ||. Must be > 0.

        eager : bool
            If True, then terminate as soon as soon as possible after Q has
            k columns OR the error drops below tol. If False, then terminate
            as soon as possible after Q has at least k columns AND the error
            drops below tol. The meaning of the phrase "as soon as possible"
            is implementation dependent. Different implementations might not
            be able to control error tolerance and might ignore this argument.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.
        """
        raise NotImplementedError()


class QB1(QBFactorizer):

    def __init__(self, rf: RangeFinder):
        self.rangefinder = rf

    def exec(self, A, k, tol, eager, rng):
        """
        Rely on a rangefinder to obtain the matrix Q for the decomposition
        A \approx Q B. Once we have Q, we construct B = Q.T @ A and return
        (Q, B). This function is agnostic to the implementation of the
        rangefinder: it might build a rank-k matrix Q all at once or construct
        successively larger matrices Q by an iterative process. We make no
        assumptions on the rangefinder's termination criteria beyond those
        listed below.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        k : int
            Target rank for Q, passed directly to the rangefinder.
            We require k <= min(A.shape).

        tol : float
            Target error tolerance ||A - Q B||. Passed to the rangefinder
            as a target error tolerance for ||A - Q Q' A||. We require tol > 0.

        eager :
            A flag used to signal preferred termination criteria for the
            rangefinder. If True, then terminate as soon as possible once
            Q.shape[1] >= k OR ||A - Q B|| <= tol. If False, then terminate
            as soon as possible after Q.shape[1] >= k AND ||A - Q B|| <= tol.
            The precise meaning of "as soon as possible" is dependent on the
            implementation of the rangefinder. Some rangefinders have no
            control over error tolerance. If implemented properly, those
            rangefinders should raise warnings if eager=False and tol < np.inf.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Output of the underlying rangefinder.

        B : ndarray
            Equal to Q.T @ A.
        """
        rng = np.random.default_rng(rng)
        Q = self.rangefinder.exec(A, k, tol, eager, rng)
        B = Q.T @ A
        return Q, B


class QB2(QBFactorizer):

    def __init__(self, rf: RangeFinder, blk: int, overwrite_a: bool):
        self.rangefinder = rf
        self.blk = blk
        self.overwrite_a = overwrite_a

    def exec(self, A, k, tol, eager, rng):
        """
        Build a QB factorization by iteratively adding columns to Q
        and rows to B. The algorithm modifies A in-place. If
        self.overwrite_a = False, then a copy of A is made at the start
        of this function call. The algorithm initializes Q, B as empty
        matrices and proceeds roughly as follows:

            cur_blk = min(k - Q.shape[1], self.blk)
            Qi = rangefinder(A, cur_blk, np.inf, True, rng)
            Bi = Qi.T @ A
            Q = np.column_stack((Q, Qi))
            B = np.row_stack((B, Bi))
            A -= Qi @ Bi

        Termination criteria are controlled by (k, tol, eager), as discussed
        below.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        k : int
            Target for the number of columns in Q. If eager=True, then Q will
            have at most k columns. If eager=True and tol=np.inf, then

        tol : float
            Target error tolerance ||A - Q B||.

        eager :
            If True, then terminate as soon as possible once Q.shape[1] == k
            OR ||A - Q B|| <= tol. If False, then terminate as soon as
            possible after Q.shape[1] >= k AND ||A - Q B|| <= tol.
            The precise meaning of "as soon as possible" is dependent on the
            implementation of the rangefinder. Some rangefinders have no
            control over error tolerance. If implemented properly, those
            rangefinders should raise warnings if eager=False and tol < np.inf.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Output of the underlying rangefinder.

        B : ndarray
            Equal to Q.T @ A.
        """
        if not self.overwrite_a:
            A = np.copy(A)
        assert k <= A.shape[0]
        assert tol > 0
        rng = np.random.default_rng(rng)
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        sq_norm_A = la.norm(A, ord='fro') ** 2
        sq_tol = tol ** 2
        blk = self.blk
        while True:
            if eager and (B.shape[0] + blk > k):
                blk = int(k - B.shape[0])  # final block
            # Standard QB, but step in to make extra sure that
            #   the columns of "Qi" are orthogonal to cols of current "Q".
            Qi = self.rangefinder.exec(A, blk, np.inf, True, rng)
            Qi = project_out(Qi, Q, as_list=False)
            Qi = la.qr(Qi, mode='economic')[0]
            Bi = Qi.T @ A
            # Update the full factorization
            Q = np.column_stack((Q, Qi))
            B = np.row_stack((B, Bi))
            A = A - Qi @ Bi
            sq_norm_A = sq_norm_A - la.norm(Bi, ord='fro') ** 2
            tol_ok = sq_norm_A <= sq_tol
            size_ok = B.shape[0] >= k
            if eager and (tol_ok or size_ok):
                break
            elif tol_ok and size_ok:
                break
        return Q, B


class QB3(QBFactorizer):

    def __init__(self, sk_op: SORS, blk: int):
        self.sk_op = sk_op
        self.blk = blk

    def exec(self, A, k, tol, eager, rng):
        assert k <= A.shape[0]
        assert tol > 0
        util.fixed_rank_warning(eager, tol, early_stop_possible=True)
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        if tol < np.inf:
            sq_norm_A = la.norm(A, ord='fro') ** 2
            sq_tol = tol**2
        rng = np.random.default_rng(rng)
        blk = self.blk
        S = self.sk_op.exec(A, k, rng)
        if not isinstance(S, np.ndarray):
            msg = """
            This implementation requires the sketching routine to return a 
            dense matrix, as represented by a numpy ndarray. We received a 
            matrix of type %s
            """ % str(type(S))
            raise RuntimeError(msg)
        G = A @ S
        H = A.T @ G
        for i in range(int(np.ceil(k/blk))):
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
            if eager and tol < np.inf:
                sq_norm_A = sq_norm_A - la.norm(Bi, ord='fro')**2
                if sq_norm_A <= sq_tol:
                    break  # early stopping
        return Q, B


###############################################################################
#      Helper functions
###############################################################################


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
