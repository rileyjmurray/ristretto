"""
Wrappers for BLAS2 and BLAS3 operations.

    https://docs.scipy.org/doc/scipy/reference/linalg.blas.html#blas-level-2-functions
    https://docs.scipy.org/doc/scipy/reference/linalg.blas.html#blas-level-3-functions

Signatures match BLAS routines. If input matrices are dense then BLAS is called directly.
If input matrices are sparse (or abstract linear operators) then we use Python operations.
"""
import scipy.linalg.blas as blas
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import spmatrix
import numpy as np
import scipy as sp

INPUT_CHECKS = False


def dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, dense):
    """
    DGEMV  performs one of the matrix-vector operations

        y := alpha*A@x + beta*y,   or   y := alpha*A.T@x + beta*y,

     where alpha and beta are scalars, x and y are vectors and A is an
     m by n matrix.

    Parameters
    ----------
    trans : string
        With 'T' or 't', we compute A.T@x. With 'N' or 'n', we compute A@x.

    m : int
        Number of rows of A

    n : int
        Number of columns of A

    alpha : float
        A double precision real

    A : Union[np.ndarray, LinearOperator, spmatrix]
        A double precision real ndarray or an abstract linear operator.
        If A is an abstract linear operator, then it is probably either a SciPy
        LinearOperator or a SciPy sparse matrix. In principle, A only needs to
        implement __matmul__ and __rmatmul__.

    lda : int
        The number of rows of A as declared in the calling (sub)program.
        This implementation requires lda=m. We keep this argument for
        compatibility with the BLAS signature for dgemv.

    x : np.ndarray
        A double precision real ndarray, with x.ndim == 1.

    incx : int
        The increment (number of double precision "slots" of memory) between
        different elements of x. This value is only used when A is an ndarray.
        If A is a LinearOperator, we require incx == 1.

    beta : float
        A double precision real

    y : np.ndarray
        A double precision real ndarray, with y.ndim == 1. Overwrites y on exit.

    incy : int
        Analogous to "incx," but for y.

    dense : bool
        If True, assume A is a numpy ndarray. If False, assume A is
        a linear operator that implements __matmul__ and __rmatmul__.

    Returns
    -------
    None
    """
    if INPUT_CHECKS:
        assert (m, n) == A.shape
        assert lda == m
        assert x.ndim == y.ndim == 1
    if dense:
        # the SciPy interface infers (m, n) from A and requires lda = m.
        blas.dgemv(alpha, A, x, beta, y, 0, incx, 0, incy, trans, overwrite_y=True)
    else:
        if INPUT_CHECKS:
            assert incx == incy == 1
        if trans == 'T' or trans == 't':
            # Don't need to call matmul on A.T; just call rmatmul on A.
            z = A.__rmatmul__(x)
        else:
            z = A.__matmul__(x)
        blas.dscal(alpha, z, n=z.size)  # z = alpha * z
        blas.dscal(beta, y, n=y.size)  # y = beta * y
        blas.daxpy(z, y, a=1.0)  # y = 1.0 * z + y
    pass


def dgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, dense):
    """
    DGEMM  performs one of the matrix-matrix operations

        C := alpha*op( A )*op( B ) + beta*C,

     where  op( X ) is one of

        op( X ) = X   or   op( X ) = X**T,

     alpha and beta are scalars, and A, B and C are matrices, with op( A )
     an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Parameters
    ----------
    transA : string
        If transA.upper() == 'T', then op(A) = A.T
        If transA.upper() == 'N', then op(A) = A.

    transA : string
        If transB.upper() == 'T', then op(B) = B.T
        If transB.upper() == 'N', then op(B) = B.

    m : int
        Number of rows in C.

    n : int
        Number of columns in C.

    k : int
        The inner dimension of the matrix-matrix product that forms C.

    alpha : float
        A double precision real

    A : Union[np.ndarray, LinearOperator, spmatrix]
        A dense double precision real ndarray or an abstract linear operator.
        If A is an abstract linear operator, then it is probably either a SciPy
        LinearOperator or a SciPy sparse matrix. In principle, A only needs to
        implement __matmul__ and __rmatmul__.

        If A is an abstract linear operator, then B must be a dense ndarray.

    lda : int
        The number of rows of A as declared in the calling (sub)program.
        This implementation requires lda to actually equal the number of rows of A.
        We keep this argument for compatibility with the BLAS signature for dgemm.

    B : Union[np.ndarray, LinearOperator, spmatrix]
        Analogous to A. If B is a linear operator, then A must be a dense ndarray.

    ldb : int
        Equals the number of rows of B.
        We keep this argument for compatibility with the BLAS signature for dgemm.

    beta : float
        A double precision real

    C : np.ndarray
        A double precision real ndarray. Overwritten on exit.

    ldc : int
        Equals the number of rows of C.
        We keep this argument for compatibility with the BLAS signature for dgemm.

    dense : bool
        If True, assume A and B are numpy ndarrays. If False, assume that A and
        B are abstract linear operators that implement __matmul__ and __rmatmul__.
        In the latter case, we call op(A)'s matmul implementation on op(B) whenever B is
        a numpy ndarray, and if B is not an ndarray, then we call op(B)'s rmatmul
        implementation on op(A). An error is raised if neither A nor B are ndarrays.


    Returns
    -------
    None
    """
    if INPUT_CHECKS:
        assert (m, n) == C.shape
        assert ldc == m
        if transA.lower() == 't':
            assert (k, m) == A.shape
            assert lda == k
        else:
            assert (m, k) == A.shape
            assert lda == m
        if transB.lower() == 't':
            assert (n, k) == B.shape
            assert ldb == n
        else:
            assert (k, n) == B.shape
            assert ldb == k
        assert isinstance(C, np.ndarray)
    if dense:
        blas.dgemm(alpha, A, B, beta, C, transA, transB, overwrite_c=True)
    else:
        if transB == 't' or transB == 'T':
            B = B.T
        if transA == 't' or transA == 'T':
            A = A.T
        if isinstance(B, np.ndarray):
            D = A.__matmul__(B)
        elif isinstance(A, np.ndarray):
            D = B.__rmatmul__(A)
        else:
            raise ValueError()
        # TODO: replace in-place * and + with appropriate BLAS call
        #   (do SciPy-blas' dscal, daxpy handle vectorization?)
        D *= alpha
        C *= beta
        C += D
    pass
