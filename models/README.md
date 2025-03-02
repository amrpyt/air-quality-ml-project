# Models Directory

This directory contains all saved models for the air quality classification and prediction project.

## Model Types

1. **Classification Models**
   - `BiLSTM_best.h5`: Best performing BiLSTM model
   - `CNN-BiLSTM_best.h5`: Best performing CNN-BiLSTM hybrid model
   - `LSTM_best.h5`: Best performing LSTM model
   - `CNN_best.h5`: Best performing CNN model
   - `DNN_best.h5`: Best performing DNN model
   - `RNN_best.h5`: Best performing RNN model

2. **Optimized Models**
   - `bilstm_pruned.h5`: Pruned BiLSTM model
   - `bilstm_quantized.tflite`: Quantized BiLSTM model in TFLite format
   - `bilstm_student.h5`: Student model from knowledge distillation

3. **Temperature & Humidity Prediction Models**
   - `temp_model_improved.h5`: Improved temperature prediction model
   - `hum_model_improved.h5`: Improved humidity prediction model

## Model Formats

- `.h5`: HDF5 format for Keras models
- `.keras`: Native Keras format
- `.tflite`: TensorFlow Lite format for optimized models

## Usage

To load and use these models:

```python
import tensorflow as tf

# Load H5 model
model = tf.keras.models.load_model('models/BiLSTM_best.h5')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/bilstm_quantized.tflite')
interpreter.allocate_tensors()
```