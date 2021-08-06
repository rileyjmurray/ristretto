import numpy as np


class SORS:
    """
    Semi-Oblivious row sketch. Given a matrix A and a positive
    integer k, generates a matrix S with k columns, for later use
    in sketching Y = A @ S. By virtue of taking linear combinations of the
    columns, the matrix S is essentially sketching the rows of A.
    """

    def __call__(self, A, k, gen):
        """
        Return a matrix S where range(S) is "reasonably" well
        aligned with the span of the top k right singular vectors
        of A. Do this while optionally incorporating information
        about A, e.g., by subspace powering. It's possible that
        we construct the matrix S without accessing A, but in that
        situation the meaning of "reasonable" is very weak.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix we'll sketch later with Y = A @ S.

        k : int
            Number of columns of S.

        gen : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.
        """
        raise NotImplementedError()


class PoweredSketchOp(SORS):
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

    def __init__(self, num_pass, pps, stabilizer, sketch_op_gen):
        # These are the same arguments as PowerRangeFinder
        self.num_pass = num_pass
        self.pps = pps
        self.stabilizer = stabilizer
        self.sketch_op_gen = sketch_op_gen

    def __call__(self, A, k, gen):
        gen = np.random.default_rng(gen)
        if self.num_pass % 2 == 1:
            S = self.sketch_op_gen(A.shape[1], k, gen)
            passes_done = 0
            q = (self.num_pass - 1) // 2
        else:
            S = A.T @ self.sketch_op_gen(A.shape[0], k, gen)
            passes_done = 1
            if self.pps == 1:
                S = self.stabilizer(S)
            q = (self.num_pass - 2) // 2
        # q is an even integer; need to compute
        #   S := (A' A)^q S
        # up to intermediate stabilization.
        while q > 0:
            S = A @ S
            passes_done += 1
            if passes_done % self.pps == 0:
                S = self.stabilizer(S)
            S = A.T @ S
            passes_done += 1
            if passes_done % self.pps == 0:
                S = self.stabilizer(S)
            q -= 1
        return S
