import numpy as np
import scipy.linalg as la
from ristretto.ballistic.randlapack.comps.powering import PoweredSketchOp, SORS

"""
Look at SciPy's 
    https://github.com/scipy/scipy/blob/master/scipy/linalg/_interpolative_backend.py
the underlying Fortran implementation accepts function handles for matvecs 
against LinearOperator objects. They branch on "dense or LinearOperator?" at 
the top-level of a given user-facing function. 
"""

#TODO: go through the functions here and have them accept "k" and
#   "tol" as target rank and target tolerance respectively. Need to
#   reconsider if "over" should be an argument to __init__ or exec.


class BaseFRID:
    """Fixed rank ID (row or column)"""

    def exec(self, A, k, over, axis, gen):
        """
        Run a rank-k RowID (axis=0) or ColumnID (axis=1) on A,
        using oversampling parameter over.

        A RowID consists of a matrix "X" and a length-k index vector "Is" so
        that A \approx X @ A[Is,:]. The rows of X must contain a
        possibly-permuted k-by-k identity matrix.

        A ColumnID consists of a matrix "Z" and a length-k index vector "Js" so
        that A \approx A[:,Js] @ Z. The columns of Z must contain a
        possibly-permuted k-by-k identity matrix.
        """
        raise NotImplementedError()


class FRID1(BaseFRID):
    """Sketch + CPQR approach to ID"""

    def __init__(self, sk_op: SORS):
        self.sk_op = sk_op

    def exec(self, A, k, over, axis, rng):
        rng = np.random.default_rng(rng)
        if axis == 0:
            Sk = self.sk_op.exec(A, k + over, rng)
            Y = A @ Sk
            X, Is = deterministic_cpqr_row_id(Y, k)
            return X, Is
        elif axis == 1:
            Sk = self.sk_op.exec(A.T, k + over, rng).T
            Y = Sk @ A
            Z, Js = deterministic_cpqr_col_id(Y, k)
            return Z, Js
        else:
            raise ValueError()


class FRID2(BaseFRID):
    """Sketch + (CPQR skeleton) + (least squares) approach to ID"""

    def __init__(self, sk_op: SORS, lstsq=la.lstsq):
        self.sk_op = sk_op
        self.lstsq = lstsq

    def exec(self, A, k, over, axis, rng):
        rng = np.random.default_rng(rng)
        if axis == 0:
            Sk = self.sk_op.exec(A, k + over, rng)
            Y = A @ Sk
            _, _, I = la.qr(Y.T, mode='economic', pivoting=True)
            Is = I[:k]
            res = self.lstsq(A[Is, :].T, A.T)  # res[0] = pinv(A[Is,:].T) @ A.T
            X = res[0].T  # X = A @ pinv(A[Is, :])
            return X, Is
        elif axis == 1:
            Sk = self.sk_op.exec(A.T, k + over, rng).T
            Y = Sk @ A
            _, _, J = la.qr(Y, mode='economic', pivoting=True)
            Js = J[:k]
            res = self.lstsq(A[:, Js], A)
            Z = res[0]
            return Z, Js
        else:
            raise ValueError()


class BaseFRDID:
    """Fixed rank Double ID"""

    def exec(self, A, k, over, rng):
        """
        Return (X, Is, Z, Js) where
            X is A.shape[0]-by-k,
            Is is an index vector of length k,
            Z is k-by-A.shape[1],
            Js is an index vector of length k,
        so that
            A \approx X @ A[Is, Js] @ Z.

        Use oversampling parameter "over" in the sketching step.
        """
        raise NotImplementedError()


class FRDID1(BaseFRDID):
    """Sketch + CPQR approach to double ID"""

    def __init__(self,  sk_op: SORS):
        self.sk_op = sk_op

    def exec(self, A, k, over, rng):
        rng = np.random.default_rng(rng)
        #TODO: start with col ID if A.shape[0] > A.shape[1]
        Sk = self.sk_op.exec(A, k + over, rng)
        Y = A @ Sk
        X, Is = deterministic_cpqr_row_id(Y, k)
        A = A[Is, :]
        Z, Js = deterministic_cpqr_col_id(A, k)
        return X, Is, Z, Js


def deterministic_cpqr_col_id(Y, k):
    # TODO: enable early-termination CPQR in LAPACK
    Q, S, J = la.qr(Y, mode='economic', pivoting=True)
    la.solve_triangular(S[:k, :k], S[:k, k:], overwrite_b=True, lower=False)
    Z = np.zeros((k, Y.shape[1]))
    Z[:, J] = np.row_stack((np.eye(k), S[:k, k:]))
    Js = J[:k]
    # Y \approx C @ Z; C = Y[:, Js]
    return Z, Js


def deterministic_cpqr_row_id(Y, k):
    # TODO: address likely memory layout inefficiency
    X, Is = deterministic_cpqr_col_id(Y.T, k)
    X = X.T
    # Y \approx X @ R; R = Y[Is, :]
    return X, Is
