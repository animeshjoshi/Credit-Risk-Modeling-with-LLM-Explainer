from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import Simulate
import ModelRisk
import Transformations


db_name = "CredRiskDB"
engine_path = "sqlite:///" + db_name + ".db"
engine = create_engine(engine_path)

X_data = pd.read_sql("SELECT * FROM CreditRiskTrainingData", engine)


y_data = pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)['Default']

# 1️⃣ Base XGBoost classifier
xgb_model = xgb.XGBClassifier(
    booster="gbtree",
    tree_method="hist",
    eval_metric="auc",
    use_label_encoder=False,
    verbosity=0
)

# 2️⃣ Hyperparameter distribution for RandomizedSearch
param_dist = {
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [200, 400, 600, 800],
    "lambda": [1, 3, 5, 7],
    "alpha": [0, 0.5, 1, 3]
}

# 3️⃣ Randomized Search CV setup
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,                  # Number of random combinations to try
    scoring="roc_auc",          # Optimize for PD ranking
    cv=3,                       # 3-fold cross-validation
    verbose=2,
    n_jobs=-1,                  # Use all CPU cores
    random_state=42
)

# 4️⃣ Fit on training data
random_search.fit(X_data, y_data)

# 5️⃣ Best hyperparameters dictionary
best_params_dict = random_search.best_params_
print("Best XGBoost hyperparameters found by RandomizedSearchCV:")
print(best_params_dict)