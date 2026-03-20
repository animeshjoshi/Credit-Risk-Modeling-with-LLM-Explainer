import pandas as pd
from sqlalchemy import create_engine

def generate_applicant_documents(df, output_path="historical_loan_data.md"):
    """
    Converts historical loan data into a markdown document
    with one chunk per applicant for RAG indexing.

    Parameters:
        df: DataFrame with applicant features, default outcome,
            and model predictions
        output_path: path to write markdown file
    """

    def interpret_pd(pd_score):
        if pd_score >= 0.75:
            return "very high default risk"
        elif pd_score >= 0.50:
            return "high default risk"
        elif pd_score >= 0.25:
            return "moderate default risk"
        else:
            return "low default risk"

    def interpret_fico(fico):
        if fico >= 750:
            return "excellent"
        elif fico >= 700:
            return "good"
        elif fico >= 650:
            return "fair"
        else:
            return "poor"

    def interpret_ltv(ltv):
        if ltv >= 0.90:
            return "very high"
        elif ltv >= 0.75:
            return "high"
        elif ltv >= 0.50:
            return "moderate"
        else:
            return "low"

    def interpret_utilization(util):
        if util >= 0.75:
            return "very high"
        elif util >= 0.50:
            return "high"
        elif util >= 0.25:
            return "moderate"
        else:
            return "low"

    def interpret_dti(dti):
        if dti >= 0.50:
            return "severely leveraged"
        elif dti >= 0.36:
            return "highly leveraged"
        elif dti >= 0.20:
            return "moderately leveraged"
        else:
            return "conservatively leveraged"

    lines = []
    lines.append("# Historical Loan Applicant Records\n")
    lines.append(
        "This document contains individual loan applicant records "
        "including demographic features, financial characteristics, "
        "engineered interaction terms, predicted default probabilities "
        "from three models, and actual default outcomes. Each record "
        "represents one applicant and can be used to identify similar "
        "historical cases when explaining predictions for new applicants.\n"
    )

    for _, row in df.iterrows():
        lines.append(f"## Applicant {row['Loan ID']}\n")

        outcome = "defaulted" if row["Default"] == 1 else "did not default"
        employed = "employed" if row["Employed"] == 1 else "unemployed"

        lines.append(
            # Identity and outcome
            f"Loan ID: {row['Loan ID']}. "
            f"This applicant {outcome} on their loan. "

            # Demographics
            f"The applicant is {row['Age']:.0f} years old and is {employed}. "
            f"Annual income is ${row['Income']:,.0f}. "

            # Debt profile
            f"Total outstanding debt is ${row['Debt']:,.0f}, "
            f"resulting in a debt to income ratio of {row['Debt to Income']:.2f} "
            f"({interpret_dti(row['Debt to Income'])}). "

            # Credit profile
            f"FICO score is {row['Fico']:.0f} ({interpret_fico(row['Fico'])} credit quality). "
            f"Risk rating is {row['Risk Rating']:.2f}. "

            # Loan characteristics
            f"Loan balance is ${row['Balance']:,.0f}. "
            f"Loan to value ratio is {row['LTV']:.2f} ({interpret_ltv(row['LTV'])}). "
            f"Credit utilization rate is {row['Utilization Rate']:.2f} "
            f"({interpret_utilization(row['Utilization Rate'])}). "

            # Interaction terms
            f"Age and income interaction: {row['Age and Income Interaction']:,.2f}. "
            f"Employment and income interaction: {row['Employment and Income Interaction']:,.2f}. "
            f"Debt and employment interaction: {row['Debt and Employment Interaction']:,.2f}. "
            f"Utilization and debt and income interaction: {row['Utilization and Debt and Income Interaction']:.4f}. "
            f"Utilization and risk rating interaction: {row['Utilization and Risk Rating Interaction']:.4f}. "
            f"Utilization and FICO interaction: {row['Utilization and Fico Interaction']:.4f}. "
            f"LTV and FICO interaction: {row['LTV Fico Interaction']:.4f}. "
            f"Risk rating and FICO interaction: {row['Risk Rating Fico Interaction']:.4f}. "
            f"FICO and income interaction: {row['Fico and Income Interaction']:,.2f}. "
            f"Balance and risk rating interaction: {row['Balance and Risk Rating Interaction']:,.2f}. "
            f"Balance and FICO interaction: {row['Balance and FICO Interaction']:,.2f}. "
            f"Balance and debt interaction: {row['Balance and Debt Interaction']:,.2f}. "

            # Model predictions
            f"Predicted default probability from logistic regression on raw features: "
            f"{row['Logistic Regression PD Model on Raw Loan Data']:.4f} "
            f"({interpret_pd(row['Logistic Regression PD Model on Raw Loan Data'])}). "
            f"Predicted default probability from logistic regression on WOE features: "
            f"{row['Logistic Regression PD Model on Weight of Evidence Loan Data']:.4f} "
            f"({interpret_pd(row['Logistic Regression PD Model on Weight of Evidence Loan Data'])}). "
            f"Predicted default probability from XGBoost: "
            f"{row['XGBoost PD Model on Raw Loan Data']:.4f} "
            f"({interpret_pd(row['XGBoost PD Model on Raw Loan Data'])}).\n"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Document written to {output_path}")

engine_path = "sqlite:///" + "CredRiskDB" + ".db"
engine = create_engine(engine_path)
LogisticConfigs =  pd.read_sql("Select * FROM CreditRiskCompleteData;", engine)
generate_applicant_documents(LogisticConfigs, 'historical_loan_data.md')