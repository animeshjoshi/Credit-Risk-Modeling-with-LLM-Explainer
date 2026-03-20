import pandas as pd
from sqlalchemy import create_engine
def generate_metrics_document(df, output_path="model_metrics.md"):
    """
    Converts a model metrics dataframe into a markdown document
    for RAG indexing.
    
    Parameters:
        df: DataFrame with columns AUC, KS Statistic, Gini Coefficient,
            Brier Score, Model Identifier
        output_path: path to write markdown file
    """

    def interpret_auc(auc):
        if auc >= 0.80:
            return "strong discrimination"
        elif auc >= 0.70:
            return "acceptable discrimination"
        else:
            return "weak discrimination"

    def interpret_ks(ks):
        if ks >= 0.50:
            return "strong separation"
        elif ks >= 0.30:
            return "acceptable separation"
        else:
            return "weak separation"

    def interpret_brier(brier):
        if brier <= 0.10:
            return "strong probability calibration"
        elif brier <= 0.20:
            return "acceptable probability calibration"
        else:
            return "weak probability calibration"

    lines = []
    lines.append("# Model Performance Metrics\n")
    lines.append(
        "This document contains performance metrics for each credit "
        "risk model. Metrics are computed on held out data and "
        "summarize each model's ability to discriminate between "
        "defaulters and non-defaulters and produce well calibrated "
        "default probability estimates.\n"
    )

    for _, row in df.iterrows():
        model_id = row["Model Identifier"]
        lines.append(f"## {model_id}\n")

        lines.append(
            f"Model: {model_id}. "
            f"AUC: {row['AUC']:.4f}, indicating {interpret_auc(row['AUC'])}. "
            f"The model correctly ranks a random defaulter above a random "
            f"non-defaulter {row['AUC']*100:.1f}% of the time. "
            f"KS Statistic: {row['KS Statistic']:.4f}, indicating "
            f"{interpret_ks(row['KS Statistic'])} between defaulter and "
            f"non-defaulter score distributions at the optimal threshold. "
            f"Gini Coefficient: {row['Gini Coefficient']:.4f}, equivalent "
            f"to an AUC of {(row['Gini Coefficient'] + 1) / 2:.4f}. "
            f"Brier Score: {row['Brier Score']:.4f}, indicating "
            f"{interpret_brier(row['Brier Score'])}. "
            f"A lower Brier Score indicates predicted probabilities are "
            f"closer to actual outcomes on average.\n"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Document written to {output_path}")

engine_path = "sqlite:///" + "CredRiskDB" + ".db"
engine = create_engine(engine_path)
LogisticConfigs =  pd.read_sql("Select * FROM ModelMetrics;", engine)
generate_metrics_document(LogisticConfigs, "allmodelmetrics.md")