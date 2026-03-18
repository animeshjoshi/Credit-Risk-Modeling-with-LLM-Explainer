import pandas as pd
from sqlalchemy import create_engine


def generate_coefficient_document(df, output_path="logistic_coefficients.md"):
    """
    Converts a logistic regression coefficient dataframe into a
    markdown interpretation document for RAG indexing.
    
    Parameters:
        df: DataFrame with columns Features, Coefficient, Std Error,
            Z Statistic, P Value, Lower CI, Upper CI, Odds Ratio,
            Model Identifier
        output_path: path to write markdown file
    """
    
    lines = []
    lines.append("# Logistic Regression Model Coefficients\n")
    lines.append(
        "This document contains the estimated coefficients for each "
        "logistic regression model. Each coefficient represents the "
        "change in log odds of default associated with a one unit "
        "increase in the corresponding feature, holding all other "
        "features constant. Confidence intervals are at the 95% level.\n"
    )

    for model_id, model_df in df.groupby("Model Identifier"):
        lines.append(f"## {model_id}\n")

        for _, row in model_df.iterrows():
            lines.append(f"### {row['Features']}\n")

            # Direction and magnitude
            direction = "increases" if row["Coefficient"] > 0 else "decreases"
            significance = (
                "statistically significant (p < 0.05)"
                if row["Significance"] ==1
                else "not statistically significant (p >= 0.05)"
            )
            ci_note = (
                "The confidence interval does not contain zero, "
                "supporting a reliable non-zero effect."
                if not (row["CI Lower"] <= 0 <= row["CI Upper"])
                else "The confidence interval contains zero, "
                "suggesting uncertainty about the direction of the effect."
            )

            lines.append(
                f"A one unit increase in {row['Features']} {direction} "
                f"the log odds of default by {abs(row['Coefficient']):.4f}. "
                f"The corresponding odds ratio is {row['Odds Ratio']:.4f}, "
                f"meaning the odds of default are multiplied by "
                f"{row['Odds Ratio']:.4f} for each unit increase. "
                f"The standard error is {row['Std Error']:.4f} and the "
                f"Z statistic is {row['Z-Statistic']:.4f}. "
                f"The p-value is {row['P-Value']:.4f}, indicating this "
                f"coefficient is {significance}. "
                f"The 95% confidence interval is "
                f"[{row['CI Lower']:.4f}, {row['CI Upper']:.4f}]. "
                f"{ci_note}\n"
            )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Document written to {output_path}")

engine_path = "sqlite:///" + "CredRiskDB" + ".db"
engine = create_engine(engine_path)
LogisticConfigs =  pd.read_sql("Select * FROM LogisticModelConfigs;", engine)
generate_coefficient_document(LogisticConfigs, "logistic_configs_document.md")