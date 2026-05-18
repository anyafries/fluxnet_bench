# Paper Experiments

Scripts to reproduce all figures and tables in the paper.

## Running instructions

All scripts must be run from the **project root** (`fluxnet_bench/`).

---

## Pilot study and benchmarking the baselines

The benchmark must already have been run to load baseline results (see the main README).

For the benchmark summaries (i.e., **Tables 1, 5–10 and the CDF plots**), run

```bash
python paper_experiments/paper_plots.py
```
**Outputs** saved to `paper_experiments/plots/`:
- `table_{aggname}_{target}.tex` — main paper LaTeX tables (median and 90th percentile RMSE)
- `table_{aggname}_{target}_supp.tex` — supplementary tables
- `cdf_{target}_{scale}.png` — CDF plots per flux target and temporal scale

---

## Quantifying the differences between the extrapolation scenarios

To compare the P(X) and P(Y|X) distribution shift between the extrapolation scenarios (i.e., **Figures 1,2,3 and Table 3**) using a domain classifier and density ratio reweighting, run

```bash
python paper_experiments/distribution_distances/analyze.py
```

**Key booleans** near the top of the file (lines ~27–56):

| Flag | Default | Effect |
|---|---|---|
| `GET_MARGINAL_METRICS` | `False` | Compute P(X) shift metrics (Accuracy-based) |
| `GET_CONDITIONAL_METRICS` | `True` | Compute P(Y\|X) shift metrics (reweighted RMSE) |
| `COMPUTE_STAT_SIG` | `True` | Run permutation/bootstrap significance tests (slow) |
| `FULL_BOOTSTRAP` | `True` | Run full bootstrap for confidence intervals (slow) |
| `MAKE_MARGINAL_PLOTS` | `False` | Create marginal histogram PDFs |
| `MAKE_CONDITIONAL_PLOTS` | `False` | Create conditional slice PDFs |

**Outputs:**
- Metric tables printed to stdout
- Optional plots (when enabled) saved to `paper_experiments/distribution_distances/plots/`

---

## ACFs

Generates autocorrelation function (ACF) plots for ET, GPP, and NEE across hourly, daily, and weekly temporal scales for four example sites (**Figure 5**).

```bash
python paper_experiments/acfs.py
```

**Outputs** saved to `paper_experiments/plots/`:
- `acf_{site}.png` — one file per site (4 total)

---

## Map and split visualization

Generate world maps of FluxNet site locations and time-series plots illustrating the train/validation/test splits:

```bash
Rscript paper_experiments/plot_regions_and_splits.R
```

**Outputs** saved to `paper_experiments/plots/`:
- `fluxnet_map.png` — world map of all FluxNet sites
- `site_split_space.png` — map showing spatial train/test split
- `site_split_ta.png` — map showing temperature-based train/test split
- `time_split.png` — combined time-series panel showing temporal split
- `time_split_{site}.png` — individual time-series for example sites
