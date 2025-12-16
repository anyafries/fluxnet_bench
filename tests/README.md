# Tests

This folder contains validation tests to ensure our refactored benchmark produces identical results to the reference QuickEval implementation.

## Test Against QuickEval

The main test script `test_vs_quickeval.py` compares our refactored code against QuickEval to ensure no bugs were introduced during refactoring.

### Quick Start

Run a basic test for GPP on daily data:

```bash
cd /path/to/fluxnet_bench
python tests/test_vs_quickeval.py --target GPP --agg daily
python tests/test_vs_quickeval.py --target GPP --agg daily-2017
python tests/test_vs_quickeval.py --target GPP --agg daily-100-2017
```

### Expected Results

From the white paper, we expect these values for Linear Regression:

**GPP:**
| Aggregation    | Mean RMSE | Median RMSE | Max RMSE |
|----------------|-----------|-------------|----------|
| daily          | 6.170     | 5.270       | 16.466   |
| daily-2017     | 7.408     | 4.533       | 28.135   |
| daily-100-2017 | 8.567     | 6.224       | 36.733   |

**Qle:**
| Aggregation    | Mean RMSE | Median RMSE | Max RMSE  |
|----------------|-----------|-------------|-----------|
| daily          | 794.965   | 499.176     | 2292.419  |
| daily-2017     | 845.933   | 644.399     | 2380.697  |
| daily-100-2017 | 948.785   | 621.377     | 3122.620  |

