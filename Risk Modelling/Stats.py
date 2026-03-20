import pandas as pd
import numpy as np
from scipy.stats import chi2
import numpy as np
import statsmodels.api as sm

def remove_correlated(X, threshold=0.95):
    corr = X.corr().abs()
    upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    to_drop = [col for col in corr.columns if any(corr.where(upper)[col] > threshold)]
    print(f"Dropping: {to_drop}")
    return X.drop(columns=to_drop)

def remove_constant(X):
    constant = X.columns[X.std() == 0].tolist()
    print(f"Dropping constant: {constant}")
    return X.drop(columns=constant)


def hosmer_lemeshow(y_true, y_pred, g=10):
    data = pd.DataFrame({'y': y_true, 'pred': y_pred})
    # Divide into g deciles
    data['decile'] = pd.qcut(data['pred'], g, labels=False)
    
    obs = data.groupby('decile')['y'].sum()
    n = data.groupby('decile')['y'].count()
    exp = data.groupby('decile')['pred'].sum()
    
    hl_stat = np.sum((obs - exp)**2 / (exp * (1 - exp/n)))
    p_value = 1 - chi2.cdf(hl_stat, g-2)
    return hl_stat, p_value