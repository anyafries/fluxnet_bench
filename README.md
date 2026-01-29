# FLUXNET Benchmark for Domain Generalization

The FLUXNET benchmark is a framework for evaluating machine learning models under distribution shift using FLUXNET ecosystem flux data. It provides standardized data preprocessing, train/test splits, and evaluation metrics to enable fair comparison of domain generalization methods.

The benchmark tests model performance on predicting **GPP (Gross Primary Productivity), ET (Evapotranspiration), and NEE (Net Ecosystem Exchange)** across different temporal and spatial splits.

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

Optional arguments:
* `--agg`: raw, daily, seasonal (default: seasonal)
* `--setting`:
    - `insite`: for a given site, split the data 80/20 (keeping time order), train on the first 80%, test on the last 20%
    - `insite-random`: for a given site, split the data 80/20 randomly
    - `loso`: train on all sites except one, evaluate on that site (slower)
    - `logo`: leave-one-group-out (balanced clusters)
* `--start`, `--stop`: which groups/sites to run the experiment on
* `--model_name`: lr, xgb (default: xgb)
* `--params`: path to CSV file with model parameters (optional)

For example, the following runs leave-one-site-out linear regression for the 5th-10th sites on seasonal data:

```bash
python run_experiment.py --agg seasonal --setting loso --start 5 --stop 10 --model_name lr
```

Some intermediate results are stored in `results/`.

## How do I add my own model?

In `models.py`:
1. Add your model class (see `LinearModel` or `XGBoostModel` as examples)
2. Add your model to `get_model(model_name, params)` function
3. Add default parameters to `get_default_params(model_name)` function

That's all! Then run:
```bash
python run_experiment.py --model_name your_model --agg seasonal --setting insite
```

See [examples/](examples/) for a complete custom model example.

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