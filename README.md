Paper Reference:
@article{ahmed2026toward,
  title={Toward realistic and lightweight intrusion detection for ADS-B: A comprehensive threat model and multi-attack machine learning evaluation},
  author={Ahmed, Waqas and Masood, Ammar and Manzoor, Jawad},
  journal={Array},
  pages={101012},
  year={2026},
  publisher={Elsevier}
}

# ADS-B Attack Detection with Classical Machine Learning

This repository contains a modular Python version of the original notebook `Final_all_models.ipynb` for ADS-B attack detection.

## Models

The pipeline trains and evaluates the three individual models implemented in the notebook:

- K-Nearest Neighbors: `KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)`
- Support Vector Machine: `SVC(kernel="rbf", C=1, gamma="scale")`
- XGBoost: `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)`

The voting ensemble section from the notebook is intentionally not included because this repository focuses on the three standalone ML models.

## Project Structure

```text
.
|-- main.py
|-- README.md
|-- requirements.txt
`-- adsb_attack_detection/
    |-- config.py
    |-- data_preprocessing.py
    |-- evaluation.py
    |-- models.py
    `-- plots.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Place `Excel_ADS_B_Dataset_344000.csv` beside `main.py`, or pass the dataset path explicitly.

```bash
python main.py --data Excel_ADS_B_Dataset_344000.csv --output-dir results
```

Optional deterministic dataframe shuffling:

```bash
python main.py --data Excel_ADS_B_Dataset_344000.csv --shuffle-random-state 42
```

Skip model pickle files and learning curves if you only need metrics:

```bash
python main.py --data Excel_ADS_B_Dataset_344000.csv --no-save-models --no-learning-curves
```

## Outputs

The script writes these files to the output directory:

- `config.json`
- `model_summary_metrics.csv`
- `model_comparison_table.csv`
- `{model}_classification_report.csv`
- `{model}_confusion_matrix.csv`
- `{model}_false_rates.csv`
- `{model}_learning_curve.png`
- `{model}_model.pkl` when model saving is enabled
- `scaler.pkl` when model saving is enabled
