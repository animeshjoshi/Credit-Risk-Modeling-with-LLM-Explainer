import pandas as pd
from sqlalchemy import create_engine

def generate_shap_document(df, output_path="shap_values.md"):
    """
    Converts a SHAP values dataframe into a markdown document
    for RAG indexing.

    Parameters:
        df: DataFrame with columns Feature, SHAP, Model Identifier
        output_path: path to write markdown file
    """

    def interpret_magnitude(shap, max_shap):
        normalized = abs(shap) / max_shap
        if normalized >= 0.75:
            return "strongly increases predicted default probability"
        elif normalized >= 0.50:
            return "moderately increases predicted default probability"
        elif normalized >= 0.25:
            return "slightly increases predicted default probability"
        else:
            return "minimally increases predicted default probability"

    lines = []
    lines.append("# Aggregated SHAP Feature Importance\n")
    lines.append(
        "This document contains mean absolute SHAP values for each "
        "feature in the XGBoost credit risk model. SHAP values measure "
        "the average contribution of each feature to predicted default "
        "probabilities across all applicants in the dataset. All features "
        "have positive mean SHAP values, indicating that each feature "
        "increases predicted default probability above the baseline on "
        "average across the portfolio. Features are ordered by SHAP value "
        "from most to least influential.\n"
    )

    for model_id, model_df in df.groupby("Model Identifier"):
        lines.append(f"## {model_id}\n")

        model_df = model_df.reindex(
            model_df["SHAP"].abs().sort_values(ascending=False).index
        )
        max_shap = model_df["SHAP"].abs().max()

        rank = 1
        for _, row in model_df.iterrows():
            lines.append(f"### {row['Feature']}\n")

            lines.append(
                f"Model: {model_id}. "
                f"Feature: {row['Feature']}. "
                f"Importance rank: {rank} out of {len(model_df)}. "
                f"Mean SHAP value: {row['SHAP']:.4f}. "
                f"This feature {interpret_magnitude(row['SHAP'], max_shap)} "
                f"relative to the baseline prediction. "
                f"Absolute SHAP value: {abs(row['SHAP']):.4f}.\n"
            )
            rank += 1

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Document written to {output_path}")

engine_path = "sqlite:///" + "CredRiskDB" + ".db"
engine = create_engine(engine_path)
LogisticConfigs =  pd.read_sql("Select * FROM shap_values;", engine)
generate_shap_document(LogisticConfigs, "XGBoost_shapley_explain.md")