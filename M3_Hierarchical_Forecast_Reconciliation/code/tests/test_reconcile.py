import unittest

import numpy as np
from scipy.sparse import csc_matrix

from reconciliation import (
    coherence_diagnostics,
    pcg_reconcile,
    reconcile_quantile_block,
)


class TinyCatalog:
    n_nodes = 9
    n_leaves = 4
    node_ids = [
        "root",
        "CA",
        "TX",
        "a",
        "b",
        "CA_a",
        "TX_a",
        "TX_b",
    ]
    leaf_nodes = np.array([5, 6, 7, 8], dtype=np.int32)
    incidence = csc_matrix(
        np.array(
            [
                [1, 1, 1, 1],  # root
                [1, 1, 0, 0],  # CA
                [0, 0, 1, 1],  # TX
                [1, 0, 1, 0],  # category a
                [0, 1, 0, 1],  # category b
                [1, 0, 0, 0],  # CA-a leaf
                [0, 1, 0, 0],  # CA-b leaf
                [0, 0, 1, 0],  # TX-a leaf
                [0, 0, 0, 1],  # TX-b leaf
            ],
            dtype=float,
        )
    )


class ReconciliationTests(unittest.TestCase):
    def test_pcg_returns_coherent_minimum_adjustment(self):
        catalog = TinyCatalog()
        # A coherent hierarchy with deliberately inconsistent aggregate forecast.
        base = np.array([10.0, 5.0, 5.0, 5.0, 5.0, 3.0, 2.0, 3.0, 2.0])[:, None]
        weights = np.ones(catalog.n_nodes)
        leaf, diagnostics = pcg_reconcile(
            catalog.incidence,
            weights,
            base,
            base[catalog.leaf_nodes],
            tolerance=1e-12,
            max_iter=100,
        )
        reconciled = np.asarray(catalog.incidence @ leaf)
        self.assertTrue(diagnostics["cg_converged"])
        np.testing.assert_allclose(reconciled[0, 0], leaf.sum(), atol=1e-9)
        np.testing.assert_allclose(
            reconciled[[1, 2], 0].sum(), reconciled[0, 0], atol=1e-9
        )
        self.assertTrue(np.all(leaf >= 0))

    def test_coherence_detects_inconsistent_aggregate(self):
        catalog = TinyCatalog()
        values = np.array(
            [20.0, 5.0, 5.0, 5.0, 5.0, 3.0, 2.0, 3.0, 2.0]
        )[:, None]
        result = coherence_diagnostics(catalog, values)
        self.assertGreaterEqual(result["max_abs_additivity_error"], 9.0)
        self.assertGreater(result["additivity_violations_gt_1e8"], 0)

    def test_external_bottom_up_quantiles_are_coherent_and_monotone(self):
        catalog = TinyCatalog()
        # Two horizons x two quantiles, laid out horizon-major.
        base = np.array(
            [
                [4.0, 5.0, 3.0, 4.0],
                [2.0, 3.0, 2.0, 3.0],
                [2.0, 3.0, 2.0, 3.0],
                [2.0, 3.0, 2.0, 3.0],
                [2.0, 3.0, 2.0, 3.0],
                [2.0, 3.0, 2.0, 3.0],
                [2.0, 3.0, 2.0, 3.0],
                [2.0, 3.0, 2.0, 3.0],
                [2.0, 3.0, 2.0, 3.0],
            ]
        )
        reconciled, _ = reconcile_quantile_block(
            catalog, "bottom_up", base, layout=(2, 2)
        )
        values = reconciled.reshape(catalog.n_nodes, 2, 2)
        self.assertTrue(np.all(np.diff(values, axis=2) >= 0))
        np.testing.assert_allclose(
            values[0, :, :],
            values[catalog.leaf_nodes, :, :].sum(axis=0),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
