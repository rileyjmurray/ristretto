import numpy as np
import scipy.linalg as la
import ristretto.ballistic.randlapack.utilities as util
import ristretto.ballistic.randlapack.comps.qb as rist_qb
import ristretto.ballistic.randlapack.comps.powering as rist_pow


class LUDecomposer:

    def exec(self, A, k, tol, over, eager, rng):
        """
        Let A be m-by-n. For some integer ell <= k, return
            Pl: an m-by-m permutation matrix,
            L: a lower-triangular matrix of shape (m, ell),
            U: an upper-triangular matrix of shape (ell, n),
            Pu: an n-by-n permutation matrix,
        so that
            A \approx Pl @ L @ U @ Pu.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate
        k : int
            The returned LU decomposition will have rank at most k.
        tol : float
            The target error used by the randomized part of the algorithm.
            When over = 0, this parameter controls ||A - Pl @ L @ U @ Pu||.
            The precise meaning of "tol" when over > 0 is implementation
            dependent.
        over : int
            The target rank used by the randomized part of the algorithm
            will be "k + over". To avoid undesired truncation, set over=0
            and increase the value of k. E.g., a configuration with over=5
            and k=20 can avoid truncation by setting k=25 and over=0.
        eager: bool
            If True, then implementations should return as soon as possible
            after either the target rank OR the target error are achieved.
            If False, then implementations should return as soon as possible
            after we reach BOTH the target rank AND the target error.
            The meaning of the phrase "as soon as possible" is implementation
            dependent. Different implementations might not be able to control
            error tolerance and might ignore this argument.
        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.
        """
        raise NotImplementedError()


class LU1(LUDecomposer):

    def __init__(self, qb: rist_qb.QBFactorizer):
        self.qb = qb

    def exec(self, A, k, tol, over, eager, rng):
        """

        Notes
        -----
        This implementation draws from ZM2020's Algorithm 3.1.
        When over > 0 (or "tol" is small and eager=False), we have to
        truncate an intermediate factorization. Our truncation approach
        is different from that of ZM2020 Algorithm 3.1.

        Specifically, ZM2020 implements truncation by taking the leading
        k columns from Q after a QB decomposition. (ZM2020 doesn't use
        QB as an explicit subroutine, but their algorithm essentially
        starts with a QB step based on a rangefinder, which in turn is
        based on subspace iteration.) That truncation method doesn't
        make sense, because the approximation A \approx QB is invariant
        under permutations that act simultaneously on the columns of Q and
        the rows of B. So for the truncation approach in ZM2020 to have
        theoretical justification we need to make detailed assumptions on
        how Q is computed.

        Rather than state those assumptions and hope the user
        provides a QB function that meets those assumptions,
        we borrow from SSAA2018 (which presents some randomized LU
        algorithms) and truncate the output of the first LU
        factorization. In the context of SSAA2018, this truncation
        strategy required the first LU factorization to be rank revealing.
        We don't use RRLU. Moreover, we have not established any
        theoretical justification for this truncation strategy in the
        larger context of this algorithm.
        """
        rng = np.random.default_rng(rng)
        Q, B = self.qb.exec(A, k, tol, eager, rng)
        # ^ We have A \approx Q B
        P1, L1, U1 = la.lu(B.T)
        # ^ We have B = U1.T @ L1.T @ P1.T
        cutoff = min(k, U1.shape[0])
        U1 = U1[:cutoff, :]  # drop rows
        L1 = L1[:, :cutoff]  # drop columns
        Y = Q @ U1.T
        P2, L2, U2 = la.lu(Y)
        # ^ We have Q B = P2 @ L2 @ (U2 @ L1.T) @ P1.T when over=0.
        Pl = P2
        Pu = P1.T
        L = L2
        U = U2 @ L1.T
        return Pl, L, U, Pu


class LU2(LUDecomposer):

    def __init__(self, sk_op: rist_pow.SORS, lstsq: la.lstsq):
        self.sk_op = sk_op
        self.lstsq = lstsq

    def exec(self, A, k, tol, over, eager, rng):
        util.fixed_rank_warning(eager, tol, early_stop_possible=False)
        rng = np.random.default_rng(rng)
        S = self.sk_op.exec(A, k + over, rng)
        Y = A @ S
        Ly, Uy, Py = util.lupt(Y)  # Y @ Py = Ly @ Uy
        if over > 0:
            Ly = Ly[:, :k]
        py = np.where(Py)[1]  # column indices
        PyA = A[py, :]  # PyA = Py @ A
        Z = self.lstsq(Ly, PyA)
        # ^ TODO: use the fact that Ly is lower-triangular. The best
        #    way to compute the pseudo-inverse might not involve least
        #    squares.
        Lz, Uz, Pz = util.lupt(Z)  # X @ Px = Lx @ Ux
        L = Ly @ Lz
        U = Uz
        # ^ Py @ A @ Pz \approx L @ U
        return Py.T, L, U, Pz.T

