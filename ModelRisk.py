import pandas as pd
import numpy as np

import statsmodels.api as sm


def logistic_pd(X, y, significance_results_path, db_path):
    
    X = sm.add_constant(X)
    y = y

    model = sm.Logit(y,X)

    result = model.fit()
    
    default_predictions = result.predict(X)


    # Extract results into a clean DataFrame
    results_df = pd.DataFrame({
        "Coefficient":   result.params,
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

    results_df.to_csv(significance_results_path)

    from sqlalchemy import create_engine
    engine_path = "sqlite:///" + db_path + ".db"
    engine = create_engine(engine_path)
    results_df.to_sql("Logistic_PD_Coefficients", engine, if_exists="replace", index=True, index_label="Feature")

    creditrisk_data = pd.read_sql("SELECT * FROM CreditRisk", engine)

    creditrisk_data['Probability of Default'] = default_predictions

    creditrisk_data.to_sql("CreditRisk", engine, if_exists="replace", index=True, index_label = "Feature")
    
    return pd.read_sql("SELECT * FROM CreditRisk", engine)



