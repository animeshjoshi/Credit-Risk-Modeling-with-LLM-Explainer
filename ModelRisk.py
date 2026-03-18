import re

import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import StandardScaler
from sqlalchemy import inspect
import statsmodels.api as sm


def logistic_pd(X, y, db_path, model_identifier, reg=True):
    
    X = sm.add_constant(X)
    y = y

    model = sm.Logit(y,X)

    if reg == False:

        result = model.fit()
    else:
        result = model.fit_regularized(method='l1', alpha=0.1, trim_mode='auto', qc_tol=1e-10)
    
    default_predictions = result.predict(X)


    # Extract results into a clean DataFrame
    results_df = pd.DataFrame({

        "Features": result.params.index,

        "Coefficient":   result.params.values,
        "Std Error":     result.bse,
        "Z-Statistic":   result.tvalues,
        "P-Value":       result.pvalues,
        "CI Lower":      result.conf_int()[0],
        "CI Upper":      result.conf_int()[1],
    })

  
    results_df["Significance"] = results_df["P-Value"] < 0.05

    # Add odds ratios
    import numpy as np
    results_df["Odds Ratio"] = np.exp(results_df["Coefficient"])

    results_df = results_df.round(4)
    results_df['Model Identifier'] = np.repeat(model_identifier, len(results_df["Odds Ratio"] ))

    from sqlalchemy import create_engine
    engine_path = "sqlite:///" + db_path + ".db"
    engine = create_engine(engine_path)

    from sqlalchemy import inspect

    inspector = inspect(engine)
    table_exists = "LogisticModelConfigs" in inspector.get_table_names()
    if table_exists:
        current_configs = pd.read_sql("SELECT * FROM LogisticModelConfigs", engine)
        results_df = pd.concat([ current_configs, results_df], ignore_index=True)
        results_df.to_sql("LogisticModelConfigs", engine, if_exists="replace", index=False)
    else:
        results_df.to_sql("LogisticModelConfigs", engine, if_exists="replace", index=False)

    creditrisk_data = pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)

    
    
    creditrisk_data[model_identifier] = default_predictions

    creditrisk_data.to_sql("CreditRiskCompleteData", engine, if_exists="replace", index=False)

    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
    auc_score = roc_auc_score(y, default_predictions)
    fpr, tpr, thresholds = roc_curve(y, default_predictions)
    ks = max(tpr-fpr)
    gini = 2*auc_score - 1
    brier = brier_score_loss(y, default_predictions)
    import Stats
   

    model_metrics = {

        'AUC': [auc_score],
        'KS Statistic':[ks],
        "Gini Coefficient": [gini],
        'Brier Score': [brier],
       
        'Model Identifier': [model_identifier]
    }

    model_metrics_df = pd.DataFrame(model_metrics)

    inspector = inspect(engine)
    table_exists = "ModelMetrics" in inspector.get_table_names()
    if table_exists:
      
        current_metrics = pd.read_sql("SELECT * FROM ModelMetrics", engine)
        model_metrics = pd.concat([model_metrics_df, current_metrics], ignore_index=True)
        model_metrics.to_sql("ModelMetrics", engine, if_exists="replace",index=False)

    else:
        model_metrics = model_metrics_df.copy()
        model_metrics.to_sql("ModelMetrics", engine, if_exists="replace", index=False)




    
    return pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)

def xgboost_pd(X, y, db_path, model_identifier):

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, classification_report

    params = {
        'subsample':        0.6,
        'n_estimators':     400,
        'min_child_weight': 5,
        'max_depth':        6,
        'learning_rate':    0.01,
        'reg_lambda':       7,      # 'lambda' is a Python keyword, xgb uses reg_lambda
        'gamma':            5,
        'colsample_bytree': 0.6,
        'reg_alpha':        0.5,    # 'alpha' → reg_alpha in xgb
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'random_state':     42
    }

    result = xgb.XGBClassifier(**params)
    result.fit(X, y)


    default_predictions = result.predict_proba(X)[:, 1]

    from sqlalchemy import create_engine
    engine_path = "sqlite:///" + db_path + ".db"
    engine = create_engine(engine_path)


    
    creditrisk_data = pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)

    
    
    creditrisk_data[model_identifier] = default_predictions

    creditrisk_data.to_sql("CreditRiskCompleteData", engine, if_exists="replace", index=False)

    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
    auc_score = roc_auc_score(y, default_predictions)
    fpr, tpr, thresholds = roc_curve(y, default_predictions)
    ks = max(tpr-fpr)
    gini = 2*auc_score - 1
    brier = brier_score_loss(y, default_predictions)
    import Stats


    model_metrics = {

        'AUC': [auc_score],
        'KS Statistic':[ks],
        "Gini Coefficient": [gini],
        'Brier Score': [brier],
  
        'Model Identifier': [model_identifier]
    }

    model_metrics_df = pd.DataFrame(model_metrics)

    inspector = inspect(engine)
    table_exists = "ModelMetrics" in inspector.get_table_names()
    if table_exists:
      
        current_metrics = pd.read_sql("SELECT * FROM ModelMetrics", engine)
        model_metrics = pd.concat([model_metrics_df, current_metrics], ignore_index=True)
        model_metrics.to_sql("ModelMetrics", engine, if_exists="replace",index=False)

    else:
        model_metrics = model_metrics_df.copy()
        model_metrics.to_sql("ModelMetrics", engine, if_exists="replace", index=False)

    import shap

    import shap
    import numpy as np

    explainer = shap.TreeExplainer(result)
    shap_values = explainer.shap_values(X)

    shap_importance = pd.DataFrame({
        "Feature": X.columns,
        "SHAP":    np.abs(shap_values).mean(axis=0)
    }).sort_values("SHAP", ascending=False)

    print(shap_importance)
    shap_df = shap_importance.copy()

    shap_df['Model Identifier'] = model_identifier

    table_exists = "XGBoostFeatureShapleyValues" in inspector.get_table_names()
    if table_exists:
        current_shaps = pd.read_sql("SELECT * FROM shap_values", engine)
        shaps = pd.concat([current_shaps, shap_df], ignore_index=True)
        shaps.to_sql("shap_values", engine, if_exists="replace", index=False)
    else:
        shap_df.to_sql("shap_values", engine, if_exists="replace", index=False)


    print("SHAP values saved to DB.")




    
    return pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)


def beta_regress_lgd(X,db_path, model_identifier, model_identifiers_list):

    import statsmodels.api as sm
    from statsmodels.formula.api import glm
    import statsmodels.formula.api as smf

    from sqlalchemy import create_engine
    engine_path = "sqlite:///" + db_path + ".db"
    engine = create_engine(engine_path)

    complete_data = pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)
    X = pd.read_sql("SELECT * FROM CreditRiskTrainingData", engine)
    X['Default'] = complete_data['Default']

    X = X[X['Default'] == 1]
    X = X.drop('Default', axis =1)

    default = complete_data[complete_data['Default'] == 1]
    default = default.drop(['Loan ID','Default'], axis=1)

    for x in model_identifiers_list:

        default = default.drop([x], axis = 1)

    default["LGD"] = default["LGD"].clip(0.001, 0.999)


    from statsmodels.othermod.betareg import BetaModel
    import re

    def clean_col(col):
        col = col.replace(" and ", "_")
        col = col.replace(" ", "_")
        col = re.sub(r'[^a-zA-Z0-9_]', '', col)
        return col

    # Clean both default (for fitting) and X (for predicting)
    default.columns = [clean_col(col) for col in default.columns]
    X.columns = [clean_col(col) for col in X.columns]

    features = [col for col in default.columns if col != "LGD"]
    formula_string = "LGD ~ " + " + ".join(features)
    scaler = StandardScaler()
    default_scaled = default.copy()
    default_scaled[features] = scaler.fit_transform(default[features])

    result = BetaModel.from_formula(formula_string, data=default_scaled).fit()

    # Now predict — X columns match the formula
   
    lgd_rate_predictions = result.predict(X)
    lgd_predictions = lgd_rate_predictions * default['Balance']



    # Extract results into a clean DataFrame
    results_df = pd.DataFrame({

        "Features": result.params.index,

        "Coefficient":   result.params.values,
        "Std Error":     result.bse,
        "Z-Statistic":   result.tvalues,
        "P-Value":       result.pvalues,
        "CI Lower":      result.conf_int()[0],
        "CI Upper":      result.conf_int()[1],
    })

  
    results_df["Significance"] = results_df["P-Value"] < 0.05

    # Add odds ratios
    import numpy as np
    results_df["Odds Ratio"] = np.exp(results_df["Coefficient"])

    results_df = results_df.round(4)
    results_df['Model Identifier'] = np.repeat(model_identifier, len(results_df["Odds Ratio"] ))

    from sqlalchemy import inspect

    inspector = inspect(engine)
    table_exists = "BetaRegressionModelConfigs" in inspector.get_table_names()
    if table_exists:
        current_configs = pd.read_sql("SELECT * FROM BetaRegressionModelConfigs", engine)
        results_df = pd.concat([results_df, current_configs], ignore_index=True)
    results_df.to_sql("BetaRegressionModelConfigs", engine, if_exists="replace", index=False)

    y_true = default['LGD']

    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
    from scipy.stats import beta
    phi = 10
    mu = np.mean(default['LGD'])
    alpha = mu*phi
    beta_param = (1-mu)*phi
    log_likelihood = np.sum(beta.logpdf(default['LGD'], alpha, beta_param))
    mae = mean_absolute_error(y_true, lgd_rate_predictions)
    rmse = root_mean_squared_error(y_true, lgd_rate_predictions)
    r2 = r2_score(y_true, lgd_rate_predictions)
    mape = mean_absolute_percentage_error(y_true, lgd_rate_predictions)

    model_metrics = {

        'Mean Absolute Error': [mae],
        'Root Mean Squared Error':[rmse],
        "R^2": [r2],
        'Log Likelihood': [log_likelihood],
        'Mean Absolute Percentage Error': [mape],
        'Model Identifier': [model_identifier]
    }

    model_metrics_df = pd.DataFrame(model_metrics)

    inspector = inspect(engine)
    table_exists = "LGDModelMetrics" in inspector.get_table_names()
    if table_exists:
      
        current_metrics = pd.read_sql("SELECT * FROM LGDModelMetrics", engine)
        model_metrics = pd.concat([model_metrics_df, current_metrics], ignore_index=True)
        model_metrics.to_sql("LGDModelMetrics", engine, if_exists="replace",index=False)

    else:
        model_metrics = model_metrics_df.copy()
        model_metrics.to_sql("LGDModelMetrics", engine, if_exists="replace", index=False)

    creditrisk_data = pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)

    
    
    creditrisk_data[model_identifier + 'Rates'] = lgd_rate_predictions
    creditrisk_data[model_identifier + 'Balances'] = lgd_predictions






