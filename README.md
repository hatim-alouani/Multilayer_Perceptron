# multilayer-perceptron

MLP built from scratch (numpy for linear algebra, matplotlib for curves only).

## Files

| file | role |
|---|---|
| `data.py` | csv loading, stratified split, standardization, one-hot |
| `network.py` | activations, losses, init, `DenseLayer`, `Network`, gradient check |
| `metrics.py` | accuracy, precision, recall, f1, confusion matrix |
| `plots.py` | learning curves |
| `split.py` | program 1 — split the dataset |
| `train.py` | program 2 — train and save |
| `predict.py` | program 3 — load and evaluate |

## Usage

```sh
python3 split.py wdbc.csv --train-size 0.8 --seed 42

# verify backprop before trusting any curve
python3 train.py --gradient-check

python3 train.py --layer 24 24 --epochs 84 --batch-size 8 \
                 --learning-rate 0.0314 --loss categoricalCrossentropy

python3 predict.py data_valid.csv --model saved_model.npy --show 10
```

## Bonuses implemented

- optimizers: `--optimizer sgd | momentum | adam`
- early stopping with best-weight restore: `--early-stopping 15`
- metrics history written to `history.json`
- precision / recall / f1 / confusion matrix
- `plots.plot_comparison()` overlays several runs on one graph
- numerical gradient checking

## Expected result

Around 97–99% validation accuracy with the defaults. If you sit at ~63% the
network is predicting the majority class and something upstream is wrong.
