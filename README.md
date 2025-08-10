# Breast Cancer Prediction System

A machine learning project that predicts whether a breast tumor is benign or malignant using the Wisconsin Breast Cancer dataset.

## 📋 Overview

This project implements a breast cancer classification system using Random Forest algorithm with scikit-learn. The system processes medical data features and predicts cancer type with high accuracy.

## 🔧 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

**Required Libraries:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms
- `joblib` - Model serialization

## 📊 Dataset Information

**Dataset:** Wisconsin Breast Cancer Dataset (`breast-cancer-wisconsin.csv`)

**Features (9 total):**
- ClumpThickness
- UniformityCellSize
- UniformityCellShape
- MarginalAdhesion
- SingleEpithelialCellSize
- BareNuclei
- BlandChromatin
- NormalNucleoli
- Mitoses

**Target Variable:** CancerType
- `2` = Benign (non-cancerous)
- `4` = Malignant (cancerous)

## 🔄 Workflow

### 1. Data Loading & Preprocessing
- Loads the CSV dataset
- Removes the CodeNumber column
- Handles missing values by replacing '?' with NaN
- Converts features to numeric format

### 2. Data Analysis
- Displays basic dataset statistics
- Shows data distribution and summary

### 3. Data Splitting
- Splits data into training (70%) and testing (30%) sets
- Uses stratified sampling for balanced classes

### 4. Model Training
- **Algorithm:** Random Forest Classifier
- **Configuration:**
  - 300 estimators (trees)
  - Median imputation for missing values
  - Parallel processing enabled
- **Pipeline:** Imputation → Random Forest Classification

### 5. Model Evaluation
- Training and testing accuracy scores
- Detailed classification report
- Confusion matrix analysis
- Feature importance visualization

### 6. Model Persistence
- Saves trained model using joblib
- Demonstrates model loading and prediction

## 🚀 Running the Project

### Prerequisites
1. Ensure you have the required dataset: `breast-cancer-wisconsin.csv`
2. Install all dependencies listed above


## 👨‍💻 Author

**Aayush Musale**  
Date: 10/08/2025
