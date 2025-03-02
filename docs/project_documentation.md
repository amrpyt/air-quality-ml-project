# Air Quality Classification and Prediction Project Documentation

## Project Overview
**Objective**: Develop a system for air quality classification and prediction using deep learning approaches with model optimization.

### Main Goals:
1. Classify air quality based on sensor data
2. Predict temperature and humidity one hour ahead
3. Optimize deep learning models for size and speed

## Data Analysis and Preprocessing

### 1. Initial Data Exploration
- **Dataset Size**: 589,876 rows × 11 columns
- **Features**:
  - Numerical (10): CO2, TVOC, PM10, PM2.5, CO, Air Quality, LDR, O3, Temp, Hum
  - Temporal (1): ts (timestamp)
- **Data Quality**:
  - No missing values
  - No duplicate rows
  - Contains negative values in CO2 and TVOC
  - Several features have extreme outliers

#### Original Data Statistics
```
Feature     Min          Max          Mean         Std
CO2         -32195.000   5000.000    596.062      683.038
TVOC        -32754.000   1098.000    50.830       465.210
PM10        2.500        17566.980   43.470       379.081
PM2.5       0.000        1537.680    18.474       42.089
CO          109.000      885.000     408.533      119.226
Air Quality 24.000       373.000     113.971      53.524
LDR         8.000        1000.000    895.097      165.805
O3          435.000      937.000     634.037      66.548
Temp        19.300       42.800      26.356       3.203
Hum         24.500       85.900      45.180       6.696
```

### 2. Data Preprocessing Steps

#### 2.1 Handling Negative Values
- **Affected Features**: CO2 and TVOC
- **Method**: Replaced negative values with median of non-negative values
- **Results**:
  - CO2: 232 negative values corrected
  - TVOC: 319 negative values corrected

#### 2.2 Outlier Handling
- **Method**: IQR (Interquartile Range) method
  - Lower bound = Q1 - 1.5 * IQR
  - Upper bound = Q3 + 1.5 * IQR
- **Results After Outlier Handling**:
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

#### 2.3 Feature Normalization
- **Method**: StandardScaler (zero mean and unit variance)
- **Impact**: All features now have comparable scales
- **Benefit**: Improves neural network training stability

#### 2.4 Class Balancing (SMOTE)
- **Original Class Distribution**:
```
Class                               Count
Good (0)                           43,768
Moderate (1)                       255,994
Unhealthy for Sensitive Groups (2) 89,972
Unhealthy (3)                      174,047
Very Unhealthy (4)                 26,038
Hazardous (5)                      57
```
- **Balanced Distribution After SMOTE**:
```
Class                               Count
All Classes                         255,994
```
- **Total Samples**: Increased from 589,876 to 1,535,964

### 3. Data Storage
- Preprocessed data saved to: `preprocessed/preprocessed_data.csv`
- Features normalized and ready for model training
- Classes balanced and encoded numerically

## Feature Analysis Results

### Correlation Analysis

#### Temperature Correlations
| Feature | Correlation |
|---------|-------------|
| O3 | -0.525606 |
| LDR | -0.309096 |
| CO | -0.183718 |
| PM10 | -0.057730 |
| PM2.5 | -0.042147 |
| TVOC | 0.019555 |
| Air Quality | 0.062345 |
| CO2 | 0.109052 |

#### Humidity Correlations
| Feature | Correlation |
|---------|-------------|
| O3 | 0.460939 |
| CO | 0.346130 |
| LDR | 0.276206 |
| Air Quality | 0.140177 |
| TVOC | 0.004280 |
| PM10 | -0.036248 |
| PM2.5 | -0.051809 |
| CO2 | -0.060488 |

### Random Forest Feature Importance

#### Temperature Prediction
| Feature | Importance |
|---------|------------|
| O3 | 0.4200 |
| LDR | 0.1697 |
| Air Quality | 0.1070 |
| CO | 0.0994 |
| PM2.5 | 0.0612 |
| CO2 | 0.0609 |
| TVOC | 0.0485 |
| PM10 | 0.0333 |

#### Humidity Prediction
| Feature | Importance |
|---------|------------|
| O3 | 0.3131 |
| LDR | 0.1732 |
| Air Quality | 0.1652 |
| CO | 0.1046 |
| PM2.5 | 0.0878 |
| TVOC | 0.0571 |
| CO2 | 0.0543 |
| PM10 | 0.0447 |

### Key Feature Analysis Findings

1. **Most Important Features**:
   - **O3 (Ozone)** is consistently the most important feature across all analyses.
   - **LDR (Light Dependent Resistor)** is the second most important feature.
   - **CO (Carbon Monoxide)** is also significant, especially for humidity prediction.

2. **Least Important Features**:
   - **PM10** consistently shows the least importance across all analyses.
   - **TVOC** also has relatively low importance for both targets.

3. **Correlation Patterns**:
   - There is a strong negative correlation between temperature and O3.
   - There is a strong positive correlation between humidity and O3.
   - Temperature and humidity themselves have a moderate negative correlation.

## Model Development

### 1. Basic Models Architecture

#### 1.1 Deep Neural Network (DNN)
```
Layer (type)                 Output Shape              Params
================================================================
Dense (256 units)           (None, 256)               2,816
BatchNormalization          (None, 256)               1,024
Dropout (0.3)              (None, 256)               0
Dense (128 units)          (None, 128)               32,896
BatchNormalization          (None, 128)               512
Dropout (0.3)              (None, 128)               0
Dense (64 units)           (None, 64)                8,256
BatchNormalization          (None, 64)                256
Dense (6 units)            (None, 6)                 390
================================================================
Total params: 46,150
```

#### 1.2 1D Convolutional Neural Network (CNN)
```
Layer (type)                 Output Shape              Params
================================================================
Conv1D (64 filters)         (None, 8, 64)             256
BatchNormalization          (None, 8, 64)             256
MaxPooling1D               (None, 4, 64)             0
Conv1D (128 filters)        (None, 2, 128)            24,704
BatchNormalization          (None, 2, 128)            512
MaxPooling1D               (None, 1, 128)            0
Flatten                    (None, 128)               0
Dense (128 units)          (None, 128)               16,512
Dropout (0.3)              (None, 128)               0
Dense (6 units)            (None, 6)                 774
================================================================
Total params: 43,014
```

#### 1.3 Simple RNN
```
Layer (type)                 Output Shape              Params
================================================================
SimpleRNN (128 units)       (None, 10, 128)          16,512
BatchNormalization          (None, 10, 128)          512
Dropout (0.3)              (None, 10, 128)          0
SimpleRNN (64 units)        (None, 64)               12,352
BatchNormalization          (None, 64)               256
Dense (6 units)            (None, 6)                 390
================================================================
Total params: 30,022
```

#### 1.4 LSTM
```
Layer (type)                 Output Shape              Params
================================================================
LSTM (128 units)            (None, 10, 128)          66,560
BatchNormalization          (None, 10, 128)          512
Dropout (0.3)              (None, 10, 128)          0
LSTM (64 units)             (None, 64)               49,408
BatchNormalization          (None, 64)               256
Dense (6 units)            (None, 6)                 390
================================================================
Total params: 117,126
```

#### 1.5 Bidirectional LSTM
```
Layer (type)                 Output Shape              Params
================================================================
Bidirectional LSTM          (None, 10, 256)          133,120
BatchNormalization          (None, 10, 256)          1,024
Dropout (0.3)              (None, 10, 256)          0
Bidirectional LSTM          (None, 128)              98,816
BatchNormalization          (None, 128)              512
Dense (6 units)            (None, 6)                 774
================================================================
Total params: 234,246
```

### 2. Hybrid Models Architecture

#### 2.1 CNN-LSTM
The CNN-LSTM hybrid model combines convolutional layers for feature extraction with LSTM layers for temporal pattern recognition:

```
Layer (type)                 Output Shape              Params
================================================================
Conv1D (64 filters)         (None, 8, 64)             256
BatchNormalization          (None, 8, 64)             256
MaxPooling1D               (None, 4, 64)             0
Conv1D (128 filters)        (None, 4, 128)            24,704
BatchNormalization          (None, 4, 128)            512
LSTM (128 units)            (None, 4, 128)            131,584
BatchNormalization          (None, 4, 128)            512
Dropout (0.3)              (None, 4, 128)            0
LSTM (64 units)             (None, 64)                49,408
BatchNormalization          (None, 64)                256
Dense (64 units)           (None, 64)                4,160
Dropout (0.3)              (None, 64)                0
Dense (6 units)            (None, 6)                 390
================================================================
Total params: 212,038
```

#### 2.2 CNN-BiLSTM
The CNN-BiLSTM hybrid model enhances the CNN-LSTM architecture with bidirectional LSTM layers:

```
Layer (type)                 Output Shape              Params
================================================================
Conv1D (64 filters)         (None, 8, 64)             256
BatchNormalization          (None, 8, 64)             256
MaxPooling1D               (None, 4, 64)             0
Conv1D (128 filters)        (None, 4, 128)            24,704
BatchNormalization          (None, 4, 128)            512
Bidirectional LSTM          (None, 4, 256)            263,168
BatchNormalization          (None, 4, 256)            1,024
Dropout (0.3)              (None, 4, 256)            0
Bidirectional LSTM          (None, 128)               98,816
BatchNormalization          (None, 128)               512
Dense (64 units)           (None, 64)                8,256
Dropout (0.3)              (None, 64)                0
Dense (6 units)            (None, 6)                 390
================================================================
Total params: 397,894
```

### 3. Enhanced Hybrid Model Improvements

The improved CNN-BiLSTM hybrid model includes:

1. **Enhanced CNN Architecture**
   - Deeper network with 3 convolutional layers (64 -> 128 -> 256 filters)
   - Added BatchNormalization and proper padding
   - Improved sequence preservation for temporal features

2. **Optimized BiLSTM Layers**
   - Dual BiLSTM layers (128 and 64 units)
   - Better long-term dependency handling
   - Strategic dropout (0.3) between layers

3. **Dense Layer Optimization**
   - Larger layers with 128 -> 64 units progression
   - Added BatchNormalization
   - Optimized dropout rates

### 4. Training Configuration
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Maximum Epochs**: 5
- **Early Stopping**: Patience = 2 epochs
- **Data Split**:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

### 5. Model Training Features
- Input Features (10): CO2, TVOC, PM10, PM2.5, CO, Air Quality, LDR, O3, Temp, Hum
- Target Classes (6): Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous
- All features are normalized (zero mean and unit variance)
- Data is balanced using SMOTE

## Model Training Results

### Performance Summary

| Model  | Test Accuracy | Test Loss | Training Time/Epoch |
|--------|--------------|-----------|-------------------|
| BiLSTM | 92.52%      | 0.1948    | ~360s            |
| CNN-BiLSTM | 92.19%  | 0.2030    | ~390s            |
| LSTM   | 92.20%      | 0.2030    | ~270s            |
| CNN-LSTM | 91.87%    | 0.2190    | ~300s            |
| CNN    | 91.48%      | 0.2190    | ~110s            |
| DNN    | 90.77%      | 0.2371    | ~65s             |
| RNN    | 90.43%      | 0.2491    | ~140s            |

### Key Findings

1. **Model Performance Ranking**:
   - BiLSTM achieved the best performance with 92.52% accuracy
   - CNN-BiLSTM was a close second with 92.19% accuracy
   - LSTM was also competitive with 92.20% accuracy
   - CNN-LSTM provided a good balance of performance and training speed
   - CNN, DNN, and RNN showed competitive but slightly lower performance

2. **Class-wise Performance**:
   - All models achieved 100% accuracy for Class 5
   - Class 3 was consistently the most challenging to classify
   - Classes 0 and 4 showed strong performance across all models

3. **Training Characteristics**:
   - All models showed consistent improvement over 5 epochs
   - No significant overfitting was observed
   - BiLSTM and CNN-BiLSTM required longer training times but delivered better results

### Model-Specific Insights

1. **BiLSTM**:
   - Best overall performance
   - Excellent precision-recall balance
   - Longest training time (~360s/epoch)

2. **CNN-BiLSTM**:
   - Very close to BiLSTM performance
   - Enhanced feature extraction capabilities
   - Slightly longer training time than BiLSTM

3. **LSTM**:
   - Very close to BiLSTM performance
   - Strong performance on all classes
   - Moderate training time (~270s/epoch)

4. **CNN-LSTM**:
   - Good balance of performance and efficiency
   - Effective feature extraction
   - Moderate training time (~300s/epoch)

5. **CNN**:
   - Good performance with faster training
   - Effective feature extraction
   - Efficient training time (~110s/epoch)

6. **DNN**:
   - Simplest architecture
   - Fastest training (~65s/epoch)
   - Competitive performance

7. **RNN**:
   - Basic sequential learning
   - Moderate training time (~140s/epoch)
   - Room for improvement

## Temperature and Humidity Prediction

### 1. Improved Model Architecture

#### Base Architecture
- **Type**: LSTM with Attention Mechanism
- **LSTM Units**: 64 (first layer), 32 (second layer)
- **Attention**: Multi-head attention with 2 heads
- **Dense Units**: 32 with ReLU activation
- **Dropout Rate**: 0.2
- **Output**: Single value (temperature or humidity)

#### Feature Engineering
- **Original Features**: O3, LDR, CO
- **Derived Features**:
  - O3/LDR ratio (interaction between top two features)
  - CO-O3 interaction (both important for humidity)
  - Time of day features (sine and cosine encoding)

### 2. Training Configuration
- **Sample Size**: 10,000 records
- **Window Size**: 60 time steps (predicts one hour ahead)
- **Epochs**: 3
- **Batch Size**: 32
- **Early Stopping**: Enabled with patience=5, monitoring validation loss

### 3. Performance Metrics

#### Initial Model Results
##### Temperature Prediction
- **MSE**: 1.0004
- **RMSE**: 1.0002
- **R²**: 0.0015
- **MAE**: 0.7448

##### Humidity Prediction
- **MSE**: 1.0003
- **RMSE**: 1.0002
- **R²**: 0.0006
- **MAE**: 0.7379

#### Optimized Model Results
##### Temperature Model
- **Final Training Loss**: 0.0234
- **Final Training MAE**: 0.1146
- **Validation Loss**: 0.0202
- **Validation MAE**: 0.1088
- **R²**: 0.8756

##### Humidity Model
- **Final Training Loss**: 0.0209
- **Final Training MAE**: 0.1090
- **Validation Loss**: 0.0190
- **Validation MAE**: 0.1022
- **R²**: 0.8912

### 4. Optimization Techniques Applied
1. **Mixed Precision Training**
   - Implemented FP16 precision for faster training
   - Used TensorFlow's mixed precision policy

2. **Feature Engineering**
   - Focused feature selection based on importance
   - Added derived interaction features
   - Implemented cyclic time encoding

3. **Model Architecture**
   - Bidirectional LSTM layers (32 -> 16 units)
   - Dropout layers (0.2) for regularization
   - Dense output layer

4. **Training Optimizations**
   - Early stopping with patience=5
   - Learning rate reduction on plateau
   - Efficient sequence creation
   - MinMaxScaler for feature normalization

## Model Optimization

### Overview
After evaluating various model architectures, the BiLSTM model emerged as the best performer with 92.52% accuracy. To improve efficiency and reduce model size while maintaining performance, we implemented several optimization techniques:

1. **Model Pruning**
   - Initial sparsity: 30%
   - Final sparsity: 70%
   - Pruning schedule: Polynomial decay
   - Target: Dense layers
   - Training duration: 5 epochs

2. **Knowledge Distillation**
   - Teacher model: Original BiLSTM
   - Student model architecture:
     - Bidirectional LSTM (64 units, return sequences)
     - Dropout (0.3)
     - Bidirectional LSTM (32 units)
     - Dense layer (64 units)
     - Output layer (6 units, softmax)
   - Temperature: 2.0
   - Alpha (distillation weight): 0.1

3. **Post-Training Quantization**
   - Format: TFLite
   - Precision: Float16
   - Optimization level: Default

### Implementation Details
The optimization process includes:

1. **Pruning Implementation**
   ```python
   pruning_params = {
       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
           initial_sparsity=0.30,
           final_sparsity=0.70,
           begin_step=0,
           end_step=len(X_train) * 5
       )
   }
   ```

2. **Knowledge Distillation**
   - Custom loss function combining:
     - Soft targets from teacher (temperature-scaled)
     - Hard targets from ground truth
   - Balanced with alpha parameter (0.1)

3. **Quantization**
   - TFLite conversion with float16 precision
   - Default optimization settings

### Optimization Results

| Model | Accuracy (%) | Model Size (MB) | Inference Time (ms) |
|-------|--------------|-----------------|---------------------|
| Original BiLSTM | 92.52 | 1.14 | 19.5 |
| Pruned BiLSTM | 92.48 | 0.42 | 12.3 |
| Quantized BiLSTM | 92.30 | 0.29 | 8.7 |
| Knowledge Distillation | 92.15 | 0.38 | 11.2 |

### Model Storage
Optimized models are saved in the following formats:
- Pruned model: `models/bilstm_pruned.keras`
- Student model: `models/bilstm_student.keras`
- Quantized model: `models/bilstm_quantized.tflite`

## Project Structure

```
project/
├── data/                         # Raw and processed data
│   └── data.csv                  # Original dataset
├── models/                       # Saved models
│   ├── BiLSTM_best.h5            # Best BiLSTM model
│   ├── CNN-BiLSTM_best.h5        # Best CNN-BiLSTM model
│   ├── bilstm_pruned.h5          # Pruned BiLSTM model
│   └── bilstm_quantized.tflite   # Quantized BiLSTM model
├── notebooks/                    # Jupyter notebooks
│   ├── step1_data_preprocessing.py
│   ├── step2_model_definitions.py
│   ├── step3_model_training.py
│   ├── step4_class3_analysis.py
│   ├── step5_feature_analysis.py
│   ├── step6_improved_model.py
│   ├── step7_model_optimization.py
│   ├── step8_transformer_model.py
│   ├── step9_ensemble_model.py
│   └── step10_bilstm_optimization.py
├── results/                      # Results and visualizations
│   ├── model_metrics/            # Performance metrics
│   └── visualizations/           # Generated plots
└── docs/                         # Documentation
    ├── project_documentation.md  # This comprehensive document
    └── ML_project_plan.md        # Original project plan
```

## Key Achievements

1. **Data Preprocessing**
   - Successfully handled outliers and normalized features
   - Created derived features to improve model performance
   - Categorized air quality according to standard guidelines
   - Extracted time-based features from timestamps

2. **Classification Models**
   - Implemented multiple deep learning architectures
   - BiLSTM achieved the best performance with 92.52% accuracy
   - CNN-BiLSTM showed excellent generalization with 92.19% accuracy
   - Optimized models for deployment with pruning and quantization

3. **Temperature & Humidity Prediction**
   - Implemented sliding window approach (window size = 60)
   - Created a BiLSTM model for accurate time series forecasting
   - Achieved good performance metrics for both temperature and humidity
   - Developed a system for hourly forecasting

4. **Model Optimization**
   - Applied various optimization techniques (pruning, quantization, knowledge distillation)
   - Reduced model size while maintaining accuracy
   - Created efficient models suitable for deployment
   - Improved inference time for real-time applications

## Conclusion and Next Steps

### Conclusion
This project successfully demonstrates the application of deep learning techniques for air quality classification and temperature/humidity prediction. The models developed show good performance and can be used for real-time monitoring and forecasting of environmental conditions.

### Next Steps

1. **Model Deployment**
   - Deploy optimized models to edge devices
   - Implement real-time prediction system
   - Create user-friendly interfaces

2. **Data Collection**
   - Collect additional data for rare classes
   - Incorporate external weather data
   - Implement continuous learning

3. **Model Improvements**
   - Explore ensemble methods for improved accuracy
   - Investigate transfer learning approaches
   - Implement online learning for model adaptation

4. **System Integration**
   - Integrate with existing monitoring systems
   - Develop APIs for third-party applications
   - Create automated alerting mechanisms