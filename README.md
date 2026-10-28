# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave, and explains why.

Built after completing Andrew Ng's *Machine Learning Specialization*, applying supervised learning, model evaluation and bias/variance diagnosis on a real dataset.

## Problem

Losing customers is expensive. If a company can spot customers who are likely to leave, it can contact them with a retention offer *before* they go. The goal is to build a model that finds as many future churners as possible.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle): 7,043 customers, 21 columns (demographics, services, contract, billing).

- About 26.6% of customers churned, so the classes are **imbalanced**.
- `TotalCharges` was stored as text. 11 rows were blank (brand-new customers with tenure 0), so they were removed, leaving 7,032 rows.

## Approach

1. **Cleaning:** fixed `TotalCharges`, dropped `customerID`, merged "No internet service" / "No phone service" into "No".
2. **Preprocessing:** Yes/No columns to 0/1, one-hot encoding for multi-category columns, standard scaling for numeric columns (fitted on the training set only).
3. **Split:** 60% train / 20% cross-validation / 20% test, stratified to keep the churn ratio equal.
4. **Models compared:** Logistic Regression, Random Forest, XGBoost, Neural Network (scikit-learn MLP).
5. **Evaluation:** precision, recall, F1 and ROC-AUC instead of accuracy alone (a model that always says "no churn" already gets about 73% accuracy).
6. **Threshold tuning:** chose the decision threshold on the CV set.
7. **Diagnosis:** learning curves and feature coefficients.
8. **Final test:** evaluated once on the untouched test set.

## Results

### Model comparison (CV set, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.797 | 0.639 | 0.540 | 0.586 | 0.834 |
| Random Forest | 0.784 | 0.638 | 0.439 | 0.520 | 0.832 |
| XGBoost | 0.790 | 0.644 | 0.473 | 0.545 | 0.835 |
| Neural Network | 0.791 | 0.628 | 0.524 | 0.571 | 0.831 |

All four models performed almost the same. The simple, interpretable **Logistic Regression** was as good as the more complex models, so it was selected.

### Threshold tuning

At the default threshold of 0.5, the model caught only about 54% of churners. Lowering the threshold to **0.3** trades some precision for much higher recall, which makes sense when losing a customer costs more than a retention call.

### Final test set results (threshold = 0.3)

| Metric | Score |
|---|---|
| ROC-AUC | 0.849 |
| Recall | 0.802 |
| Precision | 0.530 |
| F1 | 0.638 |

Confusion matrix:

|  | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actually stayed** | 767 | 266 |
| **Actually churned** | 74 | 300 |

The model catches 300 of 374 real churners (80%). The cost is 266 false alarms, i.e. customers contacted who would have stayed anyway.

## Model diagnosis

The learning curve shows training and validation ROC-AUC converging at about 0.85 with a very small gap. The model is **not overfitting**, and more data or a more complex model is unlikely to help much. Further improvement would need better features.

## What drives churn

From the logistic regression coefficients:

- **Pushes toward leaving:** fiber optic internet, electronic check payment, paperless billing, streaming services.
- **Pushes toward staying:** two-year and one-year contracts, long tenure, no internet service, tech support and online security add-ons.

Note: `tenure`, `MonthlyCharges` and `TotalCharges` are strongly related, so the individual coefficients of these three should not be over-interpreted.

**Business takeaway:** the highest-risk customer is new, on a month-to-month contract, uses fiber optic and pays by electronic check. A sensible action is to offer new fiber customers an incentive to move to a longer contract.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter

## How to run

```bash
git clone <your-repo-url>
cd churn-project
python -m venv venv
venv\Scripts\activate          # Windows
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Download the dataset from Kaggle, place the CSV in the project folder, then open `churn.ipynb` and run all cells.

## Limitations and future work

- The threshold of 0.3 was chosen by judgement; a real company would set it using the actual cost of a retention offer versus a lost customer.
- Possible improvements: feature engineering, hyperparameter tuning, probability calibration.
- Planned: customer segmentation with K-means, and a small Streamlit app.

## Author

<Your name> | <LinkedIn or email>
