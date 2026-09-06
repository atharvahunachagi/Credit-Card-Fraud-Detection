#  Credit Card Fraud Detection using Machine Learning

A Machine Learning project for detecting **fraudulent credit card transactions** using classification algorithms. The project focuses on handling highly imbalanced transaction data and comparing different models to identify fraud effectively.

##  Project Overview

The project follows this workflow:

```text
Data Preprocessing
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Class Imbalance Handling
      ↓
Model Training
      ↓
Model Evaluation
```

##  Models Used

The following classification models are implemented and compared:

* **Logistic Regression** – baseline linear model
* **Decision Tree Classifier**
* **Tuned Decision Tree**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **Tuned XGBoost**

##  Data Preprocessing

The dataset is prepared using:

* Conversion of Boolean values to integers
* **One-Hot Encoding** for categorical features
* **Binning** of transaction amounts
* Removal of unnecessary identifier and date columns
* Missing-value checking
* **StandardScaler** for feature scaling
* Stratified **80:20 train-test split**

##  Handling Class Imbalance

Fraudulent transactions represent only a small portion of the dataset. To prevent the models from being biased toward legitimate transactions, **class weighting** is used.

Methods discussed:

* Class Weights
* SMOTE
* Under-sampling

The implemented models primarily use:

```python
class_weight='balanced'
```

XGBoost uses:

```python
scale_pos_weight=62.3
```

##  Model Evaluation

Models are evaluated using:

* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

Special attention is given to **Fraud Recall**, since missing a fraudulent transaction can have a significant financial impact.

##  Hyperparameter Tuning

The Decision Tree and XGBoost models are tuned to reduce overfitting.

For example, the tuned XGBoost model uses:

```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    scale_pos_weight=62.3,
    random_state=42
)
```

The tuned XGBoost model provides the same fraud recall as the Logistic Regression baseline while producing fewer false alarms.

##  Feature Importance

Feature importance from XGBoost is analyzed to identify which transaction characteristics contribute most to fraud detection.

```python
xg_model_tuned.feature_importances_
```

##  Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── README.md
└── Credit_Card_Fraud_Detection.ipynb
```

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Jupyter Notebook
* Machine Learning

##  How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn xgboost jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Credit_Card_Fraud_Detection.ipynb
```

and run the cells sequentially.

##  Applications

This project can be used as a foundation for:

* Credit card fraud detection
* Banking transaction monitoring
* Financial risk management
* Real-time fraud alert systems

##  Future Scope

* Hyperparameter optimization
* Advanced ensemble models
* SMOTE and other imbalance techniques
* Real-time fraud detection
* Threshold optimization based on business cost
* Deployment as a fraud detection API
