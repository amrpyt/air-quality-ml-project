# Preprocessed Data Directory

This directory contains preprocessed datasets used for model training and evaluation in the air quality classification and prediction project.

## Key Files

- `preprocessed_data.csv`: Main preprocessed dataset with normalized features
- `balanced_data.csv`: Dataset after SMOTE application for class balancing
- `train_data.csv`: Training dataset (70% of balanced data)
- `val_data.csv`: Validation dataset (15% of balanced data)
- `test_data.csv`: Test dataset (15% of balanced data)

## Preprocessing Steps Applied

1. **Outlier Handling**
   - Method: IQR (Interquartile Range)
   - Lower bound = Q1 - 1.5 * IQR
   - Upper bound = Q3 + 1.5 * IQR
   - Outliers replaced with median values

2. **Feature Normalization**
   - Method: StandardScaler (zero mean and unit variance)
   - All features normalized to comparable scales

3. **Class Balancing (SMOTE)**
   - Original class distribution was imbalanced
   - SMOTE applied to create balanced dataset
   - Final dataset size: 1,535,964 rows (5× the largest class size)

4. **Feature Engineering**
   - Added derived features based on domain knowledge
   - Created time-based features from timestamps
   - Added interaction terms between important features

## Dataset Statistics After Preprocessing

```
Feature     Min      Max      Mean     Std
CO2         326.000  574.000  462.316  60.256
TVOC        22.500   82.500   51.834   14.889
PM10        2.500    17.810   7.663    3.185
PM2.5       0.000    42.200   14.772   11.306
CO          109.000  728.500  408.385  118.794
Air Quality 24.000   303.500  113.971  53.523
LDR         870.000  990.000  927.431  26.208
O3          456.500  812.500  633.920  66.190
Temp        19.300   33.450   26.235   2.839
Hum         30.950   59.350   45.057   6.261
```

## Class Distribution After SMOTE

```
Class                               Count
Good (0)                           255,994
Moderate (1)                       255,994
Unhealthy for Sensitive Groups (2) 255,994
Unhealthy (3)                      255,994
Very Unhealthy (4)                 255,994
Hazardous (5)                      255,994
```

## Usage

To load the preprocessed data:

```python
import pandas as pd

# Load preprocessed data
df = pd.read_csv('preprocessed/preprocessed_data.csv')

# Load balanced data
balanced_df = pd.read_csv('preprocessed/balanced_data.csv')

# Load train/val/test splits
train_df = pd.read_csv('preprocessed/train_data.csv')
val_df = pd.read_csv('preprocessed/val_data.csv')
test_df = pd.read_csv('preprocessed/test_data.csv')
```

For the complete preprocessing pipeline, refer to:
- `notebooks/01_data_preprocessing.ipynb`
- `notebooks/step1_data_preprocessing.py`