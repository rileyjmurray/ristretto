import numpy as np
import scipy.linalg as la
import ristretto.ballistic.randlapack.comps.qb as rist_qb


class SVDecomposer:

    def exec(self, A, k, tol, over, eager, rng):
        """
        Return U, s, Vh where, for some integer ell <= k,
            U is A.shape[0]-by-ell,
            s is a vector of length ell,
            Vh is ell-by-A.shape[1],
        so that
            A \approx U @ diag(s) @ Vh

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate
        k : int
            The returned SVD will be truncated to at most rank k.
        tol : float
            The target error used by the randomized part of the algorithm.
            When over = 0, this parameter controls ||A - U @ diag(s) @ Vh||.
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


class SVD1(SVDecomposer):

    def __init__(self, qb: rist_qb.QBFactorizer):
        self.qb = qb

    def exec(self, A, k, tol, over, eager, rng):
        rng = np.random.default_rng(rng)
        Q, B = self.qb.exec(A, k + over, tol, eager, rng)
        U, s, Vh = la.svd(B, full_matrices=False)
        if over > 0:
            cutoff = min(k, s.size)
            U = U[:, :cutoff]
            s = s[:cutoff]
            Vh = Vh[:cutoff, :]
        U = Q @ U
        return U, s, Vh
