# Predicting Annual Wages of Full-Time Workers in Texas

Machine Learning and Data Analytics (6COSC017C) — Coursework 1, Student ID 00017665.

End-to-end regression project: predict the annual wage of a full-time employee in Texas from education,
occupation, industry, job and demographic characteristics, using the **ACS 2024 1-Year PUMS** person file
(U.S. Census Bureau, public domain).

## Project structure

```
mlda-cw1/
├── data/
│   ├── raw/            # psam_p48.csv (206 MB, not in Git - see "Data" below)
│   └── processed/      # acs_tx_2024_fulltime.csv - filtered working dataset
├── notebooks/
│   └── mlda_cw1.ipynb  # full analysis: load -> EDA -> preparation -> models -> evaluation
├── src/mlda_cw1/
│   └── codes.py        # decoding tables for the PUMS numeric codes
├── app/                # Streamlit app (in progress)
├── pyproject.toml      # dependencies
├── uv.lock             # exact pinned versions (lock file)
└── requirements.txt    # same dependencies for pip / Streamlit Cloud
```

## Data

1. Download `csv_ptx.zip` from <https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/>.
2. Unzip it and place `psam_p48.csv` in `data/raw/`.

The notebook filters the raw file (292,272 people) to full-time employees aged 18–64 (92,513 people) and saves the
result to `data/processed/`. If the raw file is missing, the notebook loads the processed file instead.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs the correct Python version automatically).

```bash
git clone https://github.com/nodirsaibov05-netizen/mlda-cw1.git
cd mlda-cw1
uv sync
```

Without uv: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.

## Run the notebook

Open `notebooks/mlda_cw1.ipynb` in VS Code or Jupyter, select the `.venv` kernel and run all cells.

## Run the Streamlit app

*(Coming in a later milestone.)*

## Licence

MIT — see [LICENSE](LICENSE).
