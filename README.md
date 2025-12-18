# Heart Disease Prediction using Machine Learning

# Project Report 
https://docs.google.com/document/d/1YQBz0GkWNyhAaC0mk0jHa-NZeTOh-2j7La_383rqIq0/edit?usp=sharing

--- 

## 1. Description
Heart disease remains the world’s leading cause of death, making early diagnosis a critical public health challenge. Machine learning (ML) models can analyze clinical and demographic patient data to identify patterns indicative of cardiovascular disease before symptoms become severe.

This project focuses on building and evaluating multiple machine learning classifiers to predict heart disease risk using patient health records. The complete workflow—from data preprocessing and feature selection to model tuning and evaluation—is demonstrated. The ultimate goal is to develop an automated decision-support system that can assist clinicians in identifying high-risk patients at an early stage, potentially reducing mortality through timely intervention.

---

## 2. Need
Heart disease is one of the most significant global health concerns. In the United States alone, someone dies from cardiovascular disease approximately every 34 seconds, and globally, coronary heart disease accounts for nearly 17 million deaths annually.

These alarming statistics highlight the necessity for reliable predictive systems. Machine learning–based tools can analyze routine clinical data to flag high-risk individuals, enabling preventive care and informed clinical decision-making. Such automated systems can significantly reduce morbidity and mortality by facilitating early detection.

---

## 3. Dataset Description
The dataset consists of **1,888 records**, created by merging **five publicly available heart disease datasets**. This consolidated dataset provides a more robust and diverse foundation for training machine learning models.

### Features Overview

| Feature | Description | Type |
|------|------------|------|
| age | Age (years) | Numeric |
| sex | Sex (1 = male, 0 = female) | Categorical (Binary) |
| cp | Chest pain type (0–3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl | Categorical |
| restecg | Resting ECG results (0–2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise-induced angina | Categorical |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0–3) | Numeric |
| thal | Thalassemia type (1–3) | Categorical |
| target | Heart disease risk (1 = high, 0 = low) | Binary |

---

## 4. Framework
- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn  
- **ML Models:** Scikit-learn, XGBoost  
- **Deep Learning:** Keras (TensorFlow backend)  
- **Environment:** Google Colab  

---

## 5. Methodology

### 5.1 Data Preprocessing
- **Missing Values:**  
  - Numerical features filled with median  
  - Categorical features filled with mode  
- **Outlier Removal:**  
  - Z-score method (|Z| > 3)  
- **Encoding & Scaling:**  
  - Categorical features encoded using `LabelEncoder`  
  - Numerical features standardized using `StandardScaler`  

### 5.2 Feature Selection
- Correlation heatmap used to identify redundant features  
- Features with correlation > 0.85 removed  
- Final features selected based on absolute correlation > 0.1 with target  

---

## 6. Model Building and Tuning

### a) Logistic Regression
- Baseline linear classifier using sigmoid activation

### b) Random Forest Classifier
- Ensemble of decision trees  
- Tuned using `RandomizedSearchCV`  
- 5-fold cross-validation with ROC-AUC scoring  

### c) XGBoost Classifier
- Gradient boosting model with regularization  
- Tuned for depth, learning rate, subsampling  

### d) Artificial Neural Network (ANN)
- Implemented using Keras  
- Architecture:
  - Input layer: 128 neurons (ReLU)
  - Hidden layers: 64 and 32 neurons (ReLU)
  - Dropout: 0.4, 0.3
  - Output layer: Sigmoid
- Optimizer: Adam (lr = 0.0005)
- Early stopping applied

### e) Ensemble Model
- Soft voting ensemble combining:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - ANN

---

## 7. Confusion Matrices
Confusion matrices were generated for all models to evaluate true positives, true negatives, false positives, and false negatives, demonstrating strong classification balance for ensemble and tree-based models.

---

## 8. ANN Graphs
- Training vs Validation Accuracy  
- Training vs Validation Loss  

These plots confirm effective convergence and controlled overfitting.

---

## 9. Key Insights
- **Best Performer:** Random Forest (highest accuracy and ROC-AUC)
- **ANN:** Captured nonlinear patterns effectively but benefited from ensemble support
- **Ensemble Model:** Improved robustness and recall
- **Logistic Regression:** Served as a useful baseline but underperformed due to linear assumptions

---

## 10. Results and Discussion

### Model Performance Summary

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|------|----------|---------|-----------|--------|----------|
| Logistic Regression | 0.7735 | 0.8454 | 0.78 | 0.74 | 0.76 |
| Random Forest (Tuned) | **0.9751** | **0.9982** | 0.97 | 0.98 | 0.98 |
| XGBoost (Tuned) | 0.9613 | 0.9973 | 0.95 | 0.97 | 0.96 |
| ANN | 0.9558 | 0.9912 | 0.95 | 0.97 | 0.96 |
| Ensemble | 0.9641 | 0.9959 | 0.96 | 0.98 | 0.97 |

### Observations
- Random Forest achieved the best overall performance
- Ensemble improved stability but did not outperform RF
- Results confirm the nonlinear nature of the dataset

---

## 11. Conclusion
This project demonstrates a complete machine learning workflow for heart disease prediction. Ensemble tree-based models—especially Random Forest and XGBoost—achieved the highest accuracy and generalization. The ANN also performed strongly, while Logistic Regression served as a valuable baseline.

The findings align with existing research showing the superiority of ensemble methods for cardiovascular disease prediction. Future work could involve larger datasets, multimodal inputs (ECG images, biomarkers), and clinical validation.

---

## Future Scope
- Validation on larger and more diverse datasets  
- Integration with real-time clinical systems  
- Inclusion of imaging and biochemical markers  

---


## Project Members
- **Ananya Pahwa** 
- **Khusham Bansal** 



