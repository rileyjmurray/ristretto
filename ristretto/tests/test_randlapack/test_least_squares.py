import unittest
import numpy as np
import scipy.linalg as la
import ristretto.tests.test_randlapack.utilities as util
import ristretto.ballistic.randlapack.linops as rist_lin
import ristretto.ballistic.randlapack.drivers.least_squares as rist_lsq


def naive_bad_mat(n_rows, n_cols, scale, rng):
    rng = np.random.default_rng(rng)
    A = rng.normal(0, 1, (n_rows, n_cols))
    QA, RA = np.linalg.qr(A)
    damp = 1 / np.sqrt(1 + scale * np.arange(n_cols))
    RA *= damp
    A_bad = QA @ RA
    return A_bad


def naive_run_lstsq(seed, sap):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = 2000, 200
    A = naive_bad_mat(n_rows, n_cols, scale=5, rng=rng)
    x0 = np.random.randn(n_cols)
    b0 = A @ x0
    b = b0 + 0.05 * rng.standard_normal(n_rows)
    x_sap = sap.exec(A, b, tol=1e-8, iter_lim=40, rng=rng)
    x_np = np.linalg.lstsq(A, b, rcond=None)[0]
    error = np.linalg.norm(x_sap - x_np)
    return error


class TestSAPs(unittest.TestCase):

    SEEDS = [1, 4, 15, 31, 42]

    """
    Sketch-and-Precondition using QR
    """

    def test_sap1_srct(self):
        sap = rist_lsq.SAP1(rist_lin.srct_operator,
                            sampling_factor=3)
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap)
            errors[i] = error
        pass

    def test_sap1_gaussian(self):
        sap = rist_lsq.SAP1(rist_lin.normalized_gaussian_operator,
                            sampling_factor=2)
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap)
            errors[i] = error
        pass

    def test_sap1_sjlt(self):
        sap = rist_lsq.SAP1(rist_lin.sjlt_operator,
                            sampling_factor=3)
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap)
            errors[i] = error
        pass

    """
    Sketch-and-Precondition using SVD
    """

    def test_sap2_srct(self):
        sap = rist_lsq.SAP2(rist_lin.srct_operator,
                            sampling_factor=3,
                            smart_init=True)
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap)
            errors[i] = error
        pass

    def test_sap2_gaussian(self):
        sap = rist_lsq.SAP2(rist_lin.normalized_gaussian_operator,
                            sampling_factor=2,
                            smart_init=True)
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap)
            errors[i] = error
        sap.smart_init = False
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap)
            errors[i] = error
        pass

    def test_sap2_sjlt(self):
        sap = rist_lsq.SAP2(rist_lin.sjlt_operator,
                            sampling_factor=3,
                            smart_init=True)
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap)
            errors[i] = error
        pass


class TestSAS(unittest.TestCase):

    def test_convergence_rate_gaussian(self):
        for seed in [103, 42, 999]:
            sas = rist_lsq.SAS1(rist_lin.gaussian_operator, np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def test_convergence_rate_srct(self):
        for seed in [103, 42, 999]:
            sas = rist_lsq.SAS1(rist_lin.srct_operator, np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def test_convergence_rate_sjlt(self):
        for seed in [103, 42, 999]:
            sas = rist_lsq.SAS1(rist_lin.sjlt_operator, np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def _test_convergence_rate(self, sas, seed):
        n_rows, n_cols = 1000, 50  #  least 10x more rows than cols
        rng = np.random.default_rng(seed)
        A = naive_bad_mat(n_rows, n_cols, 5, rng)
        x0 = np.random.randn(n_cols)
        b0 = A @ x0
        b = b0 + 0.05 * rng.standard_normal(n_rows)
        x_star = la.lstsq(A, b)[0]
        errors = []
        sampling_factors = np.arange(start=1, stop=10, step=10/n_cols)
        for sf in sampling_factors:
            sas.sampling_factor = sf
            rng = np.random.default_rng(seed)
            x_ske = sas.exec(A, b, tol=0, iter_lim=1, rng=rng)
            err = la.norm(x_ske - x_star)
            errors.append(err)
        errors = np.array(errors)
        coeffs, r2 = util.loglog_fit(sampling_factors, errors)
        self.assertLessEqual(coeffs[1], -0.5)  # at least 1/sqrt(d)
        self.assertGreaterEqual(r2, 0.7)
        pass
