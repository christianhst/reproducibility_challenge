# Reproducing CTNet: Table 2, subject A01

## 1. Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

Download from the [BCI Competition IV page](https://www.bbci.de/competition/iv/) and put `A01T.gdf` and `A01E.gdf` in `BCICIV_2a_gdf/`, and the true labels `A01T.mat` and `A01E.mat` in `true_labels/`. Then:

```bash
mkdir -p mymat_raw
python preprocessing_for_2a.py
```

## 3. Train

Each model has a script in `reproduced_model_trainings/`. Example: 5 runs of ShallowConvNet, with each run's log saved:

```bash
cd reproduced_model_trainings
mkdir -p logs_shallownet/validation_1
for r in 1 2 3 4 5; do
  python -u ShallowConvNet.py 2>&1 | tee logs_shallownet/validation_1/run$r.txt
done
```

- **Validation split:** `validation_1` is the original split used for the paper, `img[number_validate:]`; `validation_2` is the corrected one, `img[-number_validate:]`. For CTNet and Conformer pass `--validation-version validation_1` or `validation_2`; for the other models edit the two `val_..._list.append(...)` lines in the script.
- **Runs:** CTNet and Conformer use the seeds recovered from the authors' notebooks (966 and 1715), so one run each. The other models use a random seed per run, so 5 runs each.

## 4. Results

```bash
python summarize_runs.py logs_shallownet/validation_1
```

This prints the mean, std and SE. Enter them in `repro_results.xlsx`, then plot:

```bash
python ../plot_figures/repro_barplot.py
```