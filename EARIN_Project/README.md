# Movie Revenue Prediction

### Dataset
https://www.kaggle.com/datasets/utkarshx27/movies-dataset
# Movie Revenue Prediction

Machine learning pipeline to predict global box office revenue from production metadata.

## Requirements

- Python 3.13
- The Movies Dataset from Kaggle, placed at `the-movies-dataset/versions/7/` relative to the script:
  - `movies_metadata.csv`
  - `credits.csv`
  - `keywords.csv`

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Dependencies

See `requirements.txt`. Main libraries used:
- `torch` — PyTorch MLP implementation
- `xgboost` — Gradient Boosting model
- `scikit-learn` — Linear Regression, Random Forest, preprocessing utilities
- `pandas` / `numpy` — Data manipulation and numerical computation
- `matplotlib` — Visualisation (optional, disabled by default)
