# Bankruptcy Prediction Using Machine Learning

This project aims to predict the likelihood of bankruptcy based on financial indicators using machine learning models. Dimensionality reduction with PCA is applied to improve model efficiency and interpretability.

---

##  Project Overview

- **Goal:** Predict if a company will go bankrupt based on its financial attributes.
- **Dataset:** Contains 6819 samples with 96 financial features.
- **ML Techniques Used:**
  - PCA (Principal Component Analysis) for dimensionality reduction
  - Classification models: Logistic Regression, SVC, Random Forest, etc.
  - Evaluation metrics: Precision, Recall, F1 Score, ROC-AUC

---

##  Project Workflow

### 1. **Data Preprocessing**
- Load and clean the dataset.
- Handle missing values if any.
- Normalize or standardize features.

### 2. **Dimensionality Reduction**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
X_train_reduced = pd.DataFrame(pca.fit_transform(X_train))
X_test_reduced = pd.DataFrame(pca.transform(X_test))
```

### 3. **Model Training and Evaluation**
- Train multiple models on reduced data.
- Evaluate using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve
  - AUC Score

### 4. **Visualization**
- Plot separate ROC curves for each algorithm.



##  Project Structure

```
├── bankruptcy.ipynb        # Main notebook with code
├── README.md               # Project documentation
└── dataset/                # Contains Bankruptcy.csv 
```

---

##  Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install -r requirements.txt
```

---

##  Conclusion

This project demonstrates the use of PCA with machine learning models to handle high-dimensional financial data and predict company bankruptcy. Model evaluation via ROC-AUC provides clarity on performance.

