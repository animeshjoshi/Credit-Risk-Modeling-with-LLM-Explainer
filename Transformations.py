def weight_of_evidence(X, y):
    import numpy as np
    import pandas as pd
    df = X.copy()
    df["__target__"] = y.values
    total_events     = y.sum()
    total_nonevents  = (y == 0).sum()
    result = pd.DataFrame(index=X.index)

    for col in X.columns:
        df["bin"] = pd.qcut(df[col], q=10, duplicates="drop")
        grouped = df.groupby("bin")["__target__"].agg(["sum", "count"])
        grouped["non_events"] = grouped["count"] - grouped["sum"]
        grouped["dist_ev"]    = grouped["sum"] / total_events
        grouped["dist_nev"]   = grouped["non_events"] / total_nonevents
        grouped["woe"]        = np.log(grouped["dist_ev"] / grouped["dist_nev"])
        grouped["woe"].replace([np.inf, -np.inf], 0, inplace=True)
        result[col] = df["bin"].map(grouped["woe"]).values

    return result