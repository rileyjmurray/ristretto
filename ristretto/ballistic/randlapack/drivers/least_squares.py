"""
Routines for (approximately) solving over-determined least squares problems

    min{ || A x - b ||_2 + reg || x ||_2 : x in R^n }.
"""
import warnings
import scipy.linalg as la
import numpy as np
from ristretto.ballistic.randlapack.comps import deterministic as de


class LstsqSolver:
    """Solver for overdetermined ordinary least-squares."""

    def exec(self, A, b, d, tol, iter_lim, rng):
        """
        Return an approximate solution to
            min{ ||A x - b||_2 : x in R^n }.

        There is no requirement that an implementation is able to control
        the error of its returned solution. Some implementations will produce
        a solution by an iterative method. We can regard non-iterative
        implementations as iterative methods that take only one step.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Tall data matrix for overdetermined ordinary least squares.

        b : ndarray
            Right-hand-side.

        d : int
            Embedding dimension. Typically, A.shape[0] >> d > A.shape[1].

        tol : float
            This parameter is only relevant for implementations that involve
            some iterative method. Those implementations must have some kind
            of error metric (e.g., backward error in the normal equations)
            for a candidate solution. If the implementation's measurement of
            error falls below tol, then it returns the candidate solution.

            If an implementation does not use an iterative method and receives
            tol > 0, then a warning will be raised.

        iter_lim : int
            We require iter_lim > 0. Typically, iter_lim << A.shape[1].
            This parameter is only relevant for implementations that involve
            some kind of iterative method. Those implementations must terminate
            after iter_lim iterations.

            If an implementation does not use an iterative method and receives
            iter_lim > 1, then a warning will be raised. The preferred argument
            for non-iterative methods is iter_lim = 1.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        x_star : ndarray
            x_star.shape == (A.shape[1],). Approximate solution to the least
            squares problem under consideration.

        Notes
        -----
        Implementations are allowed to store problem-dependent metadata
        (such as the random sketch of A) after a call to self.exec(...).
        Any information other than the return value "x_star" must be recorded
        as metadata.
        """
        raise NotImplementedError()


class SAS1(LstsqSolver):
    """A sketch-and-solve approach to overdetermined ordinary least squares.

    When constructing objects from this class, users may specify the LAPACK
    driver to be used in solving sketched least squares problems.
    """

    def __init__(self, sketch_op_gen, lapack_driver=None,
                 overwrite_sketch=True):
        self.sketch_op_gen = sketch_op_gen
        self.lapack_driver = lapack_driver
        self.overwrite_sketch = overwrite_sketch

    def exec(self, A, b, d, tol, iter_lim, rng):
        assert d >= A.shape[1] > A.shape[0]
        assert tol < np.inf
        if tol > 0:
            msg = """
            This LstsqSolver implementation cannot directly control
            approximation error. Parameter "tol" is being ignored.
            """
            warnings.warn(msg)
        if iter_lim > 1:
            msg = """
            This LstsqSolver implementation is not iterative.
            Parameter "iter_lim" is being ignored.
            """
            warnings.warn(msg)

        rng = np.random.default_rng(rng)
        S = self.sketch_op_gen(d, A.shape[0], rng)
        A_ske = S @ A
        b_ske = S @ b
        res = la.lstsq(A_ske, b_ske,
                       cond=None, overwrite_a=self.overwrite_sketch,
                       overwrite_b=True, check_finite=False,
                       lapack_driver=self.lapack_driver)
        x_ske = res[0]
        return x_ske


class SAP1(LstsqSolver):
    """A sketch-and-precondition approach to overdetermined ordinary least
    squares. This implementation uses QR to obtain the preconditioner and
    it uses LSQR for the iterative method.

    Before starting LSQR, we run a basic sketch-and-solve (for free, given
    our QR decomposition of the sketched data matrix) to obtain a solution
    x_ske. If ||A x_ske - b||_2 < ||b||_2, then we initialize LSQR at x_ske.

    This implementation assumes A is full rank.
    """

    def __init__(self, sketch_op_gen):
        self.sketch_op_gen = sketch_op_gen

    def exec(self, A, b, d, tol, iter_lim, rng):
        assert d >= A.shape[1] > A.shape[0]
        assert tol < np.inf
        rng = np.random.default_rng(rng)
        S = self.sketch_op_gen(d, A.shape[0], rng)
        A_ske = S @ A
        Q, R = la.qr(A_ske, overwrite_a=True, mode='economic')
        b_ske = S @ b
        x_ske = la.solve_triangular(R, Q.T @ b_ske, lower=False)
        x0 = None
        if np.linalg.norm(A @ x_ske - b) < np.linalg.norm(b):
            x0 = x_ske
        res = de.upper_tri_precond_lsqr(A, b, R, tol, iter_lim, x0=x0)
        x_ske = res[0]
        return x_ske


class SAP2(LstsqSolver):
    """A sketch-and-precondition approach to overdetermined ordinary least
    squares. This implementation uses the SVD to obtain the preconditioner
    and it uses LSQR for the iterative method.

    Before starting LSQR, we run a basic sketch-and-solve (for free, given
    our SVD of the sketched data matrix) to obtain a solution x_ske.
    If ||A x_ske - b||_2 < ||b||_2, then we initialize LSQR at x_ske.

    This implementation does not require that A is full-rank.
    """

    def __init__(self, sketch_op_gen, smart_init):
        self.sketch_op_gen = sketch_op_gen
        self.smart_init = smart_init

    def exec(self, A, b, d, tol, iter_lim, rng):
        assert d >= A.shape[1] > A.shape[0]
        assert tol < np.inf
        rng = np.random.default_rng(rng)
        S = self.sketch_op_gen(d, A.shape[0], rng)
        A_ske = S @ A
        # noinspection PyTupleAssignmentBalance
        U, sigma, Vh = la.svd(A_ske, overwrite_a=True, check_finite=False,
                              full_matrices=False)
        rank = np.count_nonzero(
            sigma > sigma[0] * np.min(A.shape) * np.finfo(float).eps)
        N = Vh[:rank, :].T / sigma[:rank]
        if self.smart_init:
            # This isn't necessarily preferable, because it changes the
            # norm of b, which affects termination criteria.
            b_ske = S @ b
            x_ske = N @ (U[:, :rank].T @ b_ske)
            b_remainder = b - A @ x_ske
            if la.norm(b_remainder, ord=2) < la.norm(b, ord=2):
                # x_ske is a better starting point than the zero vector.
                y_star = de.pinv_precond_lsqr(A, b_remainder, N, tol, iter_lim)
                x_star = y_star + x_ske
            else:
                # The zero vector is at least as good as x_ske.
                x_star = de.pinv_precond_lsqr(A, b, N, tol, iter_lim)
        else:
            x_star = de.pinv_precond_lsqr(A, b, N, tol, iter_lim)
        return x_star
