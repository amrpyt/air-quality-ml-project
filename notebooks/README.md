# Notebooks Directory

This directory contains all the Jupyter notebooks and Python scripts for the air quality classification and prediction project.

## Main Notebooks

1. **Data Preprocessing**
   - `01_data_preprocessing.ipynb`: Data cleaning, normalization, and preparation
   - `step1_data_preprocessing.py`: Python script version

2. **Model Development**
   - `02_model_definitions.ipynb`: Definition of all model architectures
   - `03_model_training.ipynb`: Training and evaluation of models
   - `step2_model_definitions.py`: Python script for model definitions
   - `step3_model_development.py`: Python script for model training

3. **Analysis**
   - `step4_class3_analysis.py`: Analysis of Class 3 (Unhealthy) classification
   - `step5_feature_analysis.py`: Feature importance analysis
   - `feature_analysis.py`: Detailed feature analysis script

4. **Model Improvement**
   - `step6_improved_model.py`: Implementation of improved models
   - `improved_hybrid_model.ipynb`: Hybrid model improvements
   - `hybrid_model_comparison.ipynb`: Comparison of hybrid models

5. **Model Optimization**
   - `04_model_optimization.ipynb`: Model optimization techniques
   - `step7_model_optimization.py`: Python script for model optimization
   - `step10_bilstm_optimization.py`: BiLSTM-specific optimizations

6. **Temperature & Humidity Prediction**
   - `05_temp_hum_prediction.ipynb`: Temperature and humidity prediction
   - `temp_hum_prediction_improved.py`: Improved prediction script
   - `step8_transformer_model.py`: Transformer model for prediction
   - `step9_ensemble_model.py`: Ensemble approach for prediction

7. **Complete Pipelines**
   - `main_pipeline.ipynb`: Complete pipeline from data to results
   - `air_quality_project_complete.ipynb`: Comprehensive notebook

## Usage

To run these notebooks:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Start with data preprocessing:
   ```bash
   jupyter notebook 01_data_preprocessing.ipynb
   ```

3. Follow the numbered sequence for a step-by-step approach, or use the complete pipeline notebooks for an end-to-end solution.

## Python Scripts

The `.py` files are standalone Python scripts that can be run directly:

```bash
python step1_data_preprocessing.py
```

These scripts implement the same functionality as the notebooks but are designed for batch processing or integration into larger systems.