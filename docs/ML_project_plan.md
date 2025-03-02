### **Project Title:**  
**Air Quality Classification and Prediction with Deep Learning & Model Optimization**  

### **Objective:**  
1. **Classify air quality** based on sensor data.  
2. **Predict temperature and humidity** one hour ahead using a sliding window approach.  
3. **Optimize deep learning models** to reduce size and improve speed without compromising accuracy.  

---

### **Dataset Details:**  
- The dataset is provided in **CSV format** and contains air quality-related features, including **temperature (temp), humidity (hum), and air quality indicators**.  
- The target variable for classification is **"air quality"**.  

---

### **Step 1: Data Preprocessing**  
Perform the following preprocessing steps:  
1. **Outlier Handling:** Detect and replace outliers using appropriate methods (mean, median, etc.). **Do not delete data.**  
2. **Normalization:** Apply feature scaling techniques to standardize the dataset.  
3. **Class Balancing:** Use **SMOTE** to balance the dataset while ensuring the final dataset size is **5× the largest class size** (e.g., 1,535,916 rows if the largest class is 255,994).  

---

### **Step 2: Air Quality Classification**  
- **Train the following deep learning models** for classification:  
  - **Basic Models:** 1D-CNN, RNN, DNN, LSTM, BiLSTM  
  - **Hybrid Models:** CNN-LSTM, CNN-BiLSTM  
- **Categorization of Air Quality:** Implement the following function to label the air quality column:  

  ```python
  def categorize_air_quality(value):
      if value <= 50:
          return 'Good'
      elif 51 <= value <= 100:
          return 'Moderate'
      elif 101 <= value <= 150:
          return 'Unhealthy for Sensitive Groups'
      elif 151 <= value <= 200:
          return 'Unhealthy'
      elif 201 <= value <= 300:
          return 'Very Unhealthy'
      else:
          return 'Hazardous'
  ```
- **Evaluation Metrics:** Use **Accuracy, Recall, Precision, and F1-score**.  

---

### **Step 3: Model Optimization**  
1. **Optimize the best-performing model** using:  
   - **Pruning**  
   - **Quantization**  
   - **Clustering**  
   - **Weight Clipping**  
   - **Knowledge Distillation**  
   - **Post-Training Quantization**  
2. **Objective of Optimization:**  
   - The optimized model should have **similar or slightly better accuracy** but **fewer parameters and improved speed**.  
   - Hybrid models should have **higher accuracy and more parameters before optimization**, and **fewer parameters after optimization** while maintaining accuracy.  

---

### **Step 4: Temperature & Humidity Prediction**  
- **Use a sliding window approach (window = 60)** to predict temperature and humidity one hour ahead.  
- **Evaluation Metrics:** MSE, RMSE, R², MAE.  

---

### **Step 5: Visualization & Reporting**  
Generate **plots, tables, and reports** comparing:  
1. **Before & After Outlier Removal**  
2. **Before & After SMOTE Application**  
3. **Accuracy & Loss Curves** for training and validation  
4. **Parameter Reduction After Optimization**  

---

### **Execution & Deliverables:**  
- Implement the entire project on **Google Colab**.  
- Provide clean, modular, and well-documented code.  
- Ensure the final optimized model has **fewer parameters** than before.  
- Submit the final code along with a summary report.  
- Be prepared for **minor modifications** if required by the client.