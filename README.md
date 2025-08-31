# FLUXNET benchmark

The FLUXNET benchmark is a framework for evaluating machine learning models on FLUXNET ecosystem data. It standardises preprocessing and provides common experimental setups (random split per site, temporal holdout, leave-one-site-out) to ensure fair comparison. The benchmark is designed to test how well models generalize in predicting GPP across sites and time.

Information about the FLUXNET data: https://pad.gwdg.de/s/yuCtk9fj5

## Required packages

```python
python3 -m venv fluxnet_bench_venv
source fluxnet_bench_venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Preprocessing data

The data needs to be copied into the main directory in the "data" folder.

Run the following to obtain the cleaned raw, daily, and seasonal datasets. (This takes a while.)

```
python3 preprocessing.py
```

A folder `data_cleaned` with the aggregated datasets will be created.

## Types of experiments

Run an experiment, i.e., train the model and test on unseen data.

```
python3 run_experiment.py
```

Optional arguments
* `--agg`: raw, daily, seasonal
* `--setting`: 
    - insite: for a given site, split the data 80/20 (keeping time order), train on the first 80%, test on the last 20%
    - insite-random: for a given site, split the data 80/20 randomly
    - loso: train on all sites except one, evaluate on that site (slower)
* `--start`, `--stop`: which groups to run the experiment on
* `--model_name`: lr, xgb 

For example, the following runs leave-one-site-out linear regression for the 5th-10th sites on seasonal data: 

```
python3 run_experiment.py --agg seasonal --setting loso --start 5 --stop 10 --model lr
```

Some intermediate results are stored in `results/`.

## How do I add my own model?

In `run_experiment.py`:
1. Line 27: add your model to the `get_model(model, params={})`. 
2. Line 43: add the model parameters to `get_default_params(model)` (ignore cv argument, not implemented yet)

That's all! Then run 
```
python3 run_experiment.py --model your_model_name
```

## Evaluation

To compare any experiments with the given arguments: 

```
python3 eval.py
``` 

* `----agg`: raw, daily, seasonal
* `--setting`: insite, loso 

Output plots will appear in `results/plots_tmp`. There will also be a table of results printed to the terminal.

For example, the following evaluates any sites that have been evaluated at the seasonal aggregation in the leave-one-site-out setting:

```
python3 eval.py --agg seasonal --setting loso
```

# References

Pastorello, G. et al. (2017) ‘The FLUXNET2015 dataset: The longest record of global carbon, water, and energy fluxes is updated’, Eos, 98.

Pastorello, G. et al. (2020) ‘The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data’, Scientific Data, 7(1), p. 225. Available at: https://doi.org/10.1038/s41597-020-0534-3.