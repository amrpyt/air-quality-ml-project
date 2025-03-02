# Results Directory

This directory contains all results, metrics, and visualizations generated during the air quality classification and prediction project.

## Directory Structure

```
results/
├── model_metrics/            # Performance metrics for all models
│   ├── classification/       # Classification model metrics
│   └── prediction/           # Prediction model metrics
└── visualizations/           # Generated plots and visualizations
    ├── data_exploration/     # Data exploration visualizations
    ├── model_performance/    # Model performance visualizations
    ├── feature_analysis/     # Feature importance visualizations
    └── predictions/          # Prediction visualizations
```

## Key Files

### Classification Metrics
- `classification_metrics.pkl`: Metrics for all classification models
- `confusion_matrices.pkl`: Confusion matrices for all models
- `class_wise_metrics.csv`: Class-wise precision, recall, and F1-scores

### Prediction Metrics
- `temp_hum_metrics_improved.pkl`: Metrics for temperature and humidity prediction
- `temp_hum_prediction_metrics_final.pkl`: Final metrics after optimization

### Visualizations
- `temp_hum_predictions_detailed.png`: Detailed temperature and humidity predictions
- `model_comparison.png`: Comparison of different model architectures
- `feature_importance.png`: Feature importance visualizations
- `hourly_predictions_forecast.png`: Hourly prediction forecasts
- `project_summary.png`: Overall project summary visualization

## Usage

To load and analyze the metrics:

```python
import pickle

# Load metrics
with open('results/model_metrics/classification/classification_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Access specific metrics
accuracy = metrics['BiLSTM']['accuracy']
precision = metrics['BiLSTM']['precision']
```

To display visualizations, simply open the PNG files in any image viewer or load them in a Jupyter notebook:

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display visualization
img = mpimg.imread('results/visualizations/model_performance/model_comparison.png')
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```