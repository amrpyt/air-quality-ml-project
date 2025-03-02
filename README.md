# Air Quality Analysis and Prediction Project

## Project Overview
This project implements air quality classification and temperature/humidity prediction using various deep learning models, including hybrid architectures.

## Key Features
- Air quality classification with 6 categories
- Temperature and humidity prediction
- Hybrid model optimization
- Time series analysis with sliding window

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Project Structure
```
ML-project/
├── data/             # Dataset files
├── docs/             # Documentation
├── models/           # Saved models
├── notebooks/        # Jupyter notebooks
├── preprocessed/     # Preprocessed data
├── results/          # Analysis results
└── src/             # Source code
```

## Running the Project

### Option 1: Complete Pipeline
Run the main pipeline notebook that executes all steps:
```bash
jupyter notebook notebooks/main_pipeline.ipynb
```

This notebook includes:
1. Data preprocessing and visualization
2. Model training (basic and hybrid)
3. Model optimization
4. Time series prediction
5. Results comparison

### Option 2: Step by Step

1. **Data Preprocessing:**
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```
- Outlier detection and handling
- Data normalization
- SMOTE balancing

2. **Model Training:**
```bash
jupyter notebook notebooks/02_model_definitions.ipynb
jupyter notebook notebooks/03_model_training.ipynb
```
- Basic models (1D-CNN, RNN, DNN, LSTM, BiLSTM)
- Hybrid models (CNN-LSTM, CNN-BiLSTM)

3. **Model Optimization:**
```bash
jupyter notebook notebooks/04_model_optimization.ipynb
```
- Pruning
- Weight clipping
- Performance comparison

4. **Time Series Prediction:**
```bash
jupyter notebook notebooks/05_temp_hum_prediction.ipynb
```
- Temperature and humidity prediction
- 60-timestep sliding window

## Model Architecture
- **Input Shape:** (timesteps, features)
- **Basic Models:** 1D-CNN, RNN, DNN, LSTM, BiLSTM
- **Hybrid Models:** CNN-LSTM, CNN-BiLSTM
- **Optimized Architecture:** Reduced parameters with maintained accuracy

## Configuration
Model parameters and settings can be adjusted in `src/config.py`:
- Window size
- Model architecture
- Training parameters
- Optimization settings

## Results
Results are saved in the following locations:
- Trained models: `models/`
- Performance metrics: `results/`
- Visualizations: Generated in notebooks

## Performance Metrics
The project evaluates models using:
- Classification: Accuracy, Precision, Recall, F1-score
- Regression: MSE, RMSE, R², MAE
- Model Efficiency: Parameter count, inference time

## Documentation
- Full technical details: `docs/project_documentation.md`
- Model requirements: `docs/ML_project_plan.md`

## Troubleshooting
If you encounter issues:
1. Check Python version (3.8+ recommended)
2. Verify all dependencies are installed
3. Ensure data files are in correct locations
4. Check GPU availability for faster training

## License
This project is licensed under the MIT License - see the LICENSE file for details.