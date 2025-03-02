# Data Directory

This directory contains the raw dataset used for the air quality classification and prediction project.

## Dataset Description

The main dataset file is `data.csv`, which contains air quality sensor readings with the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| CO2 | Carbon Dioxide concentration | ppm |
| TVOC | Total Volatile Organic Compounds | ppb |
| PM10 | Particulate Matter (≤10μm) | μg/m³ |
| PM2.5 | Particulate Matter (≤2.5μm) | μg/m³ |
| CO | Carbon Monoxide | ppb |
| Air Quality | Air Quality Index | - |
| LDR | Light Dependent Resistor reading | - |
| O3 | Ozone concentration | ppb |
| Temp | Temperature | °C |
| Hum | Humidity | % |
| ts | Timestamp | - |

## Dataset Statistics

- **Size**: 589,876 rows × 11 columns
- **Time Period**: Continuous readings with 1-minute intervals
- **Missing Values**: None
- **Outliers**: Present in CO2, TVOC, PM10, and PM2.5 columns

## Air Quality Categories

The Air Quality values are categorized as follows:

| Range | Category |
|-------|----------|
| ≤ 50 | Good |
| 51-100 | Moderate |
| 101-150 | Unhealthy for Sensitive Groups |
| 151-200 | Unhealthy |
| 201-300 | Very Unhealthy |
| > 300 | Hazardous |

## Usage

To load the dataset:

```python
import pandas as pd

# Load data with optimized dtypes to reduce memory usage
dtypes = {
    'CO2': 'float32',
    'TVOC': 'float32',
    'PM10': 'float32',
    'PM2.5': 'float32',
    'CO': 'float32',
    'Air Quality': 'float32',
    'LDR': 'float32',
    'O3': 'float32',
    'Temp': 'float32',
    'Hum': 'float32',
    'ts': 'str'
}

# Load data (optionally with a sample size)
df = pd.read_csv('data/data.csv', dtype=dtypes)

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

## Data Preprocessing

For preprocessing steps, refer to the notebooks in the `notebooks` directory:
- `01_data_preprocessing.ipynb`: Complete preprocessing pipeline
- `step1_data_preprocessing.py`: Python script version