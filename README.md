# Predictive Modeling and Optimization in Logistics Systems

Week 4 internship project: forecast delivery time (minutes) with machine learning and use the model to optimize logistics operations.

## What this project does
1. **Simulates** a 5,000-delivery logistics dataset (distance, traffic, weather, vehicle, warehouse, stops, driver experience, etc.).
2. **Prepares** the data with a scikit-learn Pipeline (median imputation, one-hot encoding, scaling, engineered features).
3. **Compares** Linear, Ridge, Decision Tree, Random Forest and Gradient Boosting models using 5-fold cross-validation.
4. **Tunes** Gradient Boosting with GridSearchCV and evaluates on a held-out test set (RMSE, MAE, R2).
5. **Optimizes** operations using the model: Hungarian-algorithm driver assignment, cost-minimizing vehicle selection, peak-hour rescheduling and model-based ETA promises.

## Results (test set)
| Model | RMSE (min) | MAE (min) | R2 |
|---|---|---|---|
| Linear Regression | 12.95 | 9.28 | 0.919 |
| Random Forest | 9.76 | 7.50 | 0.954 |
| Gradient Boosting (tuned) | 6.46 | 4.90 | 0.980 |

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/model_training.py
python src/optimization.py
```

## Structure
```
data/logistics_deliveries.csv   simulated dataset
src/generate_data.py            data simulation
src/model_training.py           preprocessing, CV, tuning, evaluation, charts
src/optimization.py             optimization strategies
outputs/                        charts, trained model, metrics, Word report
```

The full write-up is in `outputs/Week4_Predictive_Modeling_Logistics_Report.docx`.
