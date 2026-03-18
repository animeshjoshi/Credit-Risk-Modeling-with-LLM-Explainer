import pandas as pd
import numpy as np


def simulate_risk_data(sample_size, db_name):
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    N = sample_size


    age = np.random.normal(loc=45, scale=13, size=N).clip(18, 87).astype(int)

    peak_age = 60

    income_base = -70 * (age - peak_age)**2 + 83000
    noise = np.random.lognormal(mean=0, sigma=0.35, size=N)
    income = (income_base * noise).clip(15000, 400000).round(2)

    # Age-conditioned unemployment probability
    unemployment_prob = 0.04 + 0.03 * ((age < 25) | (age > 62)).astype(float)
    employed = np.random.binomial(1, 1 - unemployment_prob, N)

    # Age parabola for debt — peaks around 55
    debt_age_factor = (-150 * (age - 55)**2 + 120000).clip(20000)

    debt_income_factor = -0.2 * income


    # Combine and add lognormal noise
    debt_base = 0.5 * debt_age_factor  + 0.5 * debt_income_factor
    noise = np.random.lognormal(mean=0, sigma=0.45, size=N)
    debt = (debt_base * noise).clip(0, 800000).round(2)

    # Unemployed borrowers get a debt bump — accumulated debt relative to fallen income
    debt = np.where(employed == 0, debt * np.random.uniform(1.1, 1.4, N), debt)
    debt = debt.clip(0, 800000).round(2)


    fico_base = 500 + (np.log(income) - np.log(15_000)) / (np.log(400_000) - np.log(15_000)) * 550
    fico_noise = np.random.normal(0, 40, N)
    fico = (fico_base + fico_noise).clip(300, 850).astype(int) - (45 * (1-employed)) - (0.0005 * debt)
    # Base utilization — younger and lower income = higher utilization
    age_effect = 0.4 - 0.003 * (age - 25).clip(0)  # decreases with age
    income_effect = 0.3 - 0.0000008 * income  # decreases with income
    employment_effect = np.where(employed == 0, 0.15, 0.0)  # unemployed draw down credit

    utilization_base = age_effect + income_effect + employment_effect

    # Lognormal noise — utilization is very behavioral
    noise = np.random.lognormal(mean=0, sigma=0.4, size=N)
    utilization = (utilization_base * noise).clip(0.01, 1.0).round(4)

    # FICO penalty for high utilization
    # 0-30%: no penalty
    # 30-60%: moderate penalty up to -50 pts
    # 60-90%: heavy penalty up to -100 pts
    # 90-100%: severe penalty up to -150 pts
    utilization_penalty = np.select(
        [
            utilization <= 0.30,
            (utilization > 0.30) & (utilization <= 0.60),
            (utilization > 0.60) & (utilization <= 0.90),
            utilization > 0.90
        ],
        [
            0,
            -50  * ((utilization - 0.30) / 0.30),
            -50  - 50 * ((utilization - 0.60) / 0.30),
            -100 - 50 * ((utilization - 0.90) / 0.10)
        ]
    )

    # Apply penalty to existing FICO
    fico = (fico + utilization_penalty).clip(300, 850).astype(int)

    # Base LTV — income and age are primary drivers
    income_effect = 0.8 - 0.0000008 * income        # higher income = lower LTV
    age_effect = -0.003 * (age - 25).clip(0)         # older = more paid down = lower LTV
    utilization_effect = 0.15 * utilization          # stretched borrowers = higher LTV
    employment_effect = np.where(employed == 0, 0.08, 0.0)  # unemployed = higher LTV

    ltv_base = income_effect + age_effect + utilization_effect + employment_effect

    dti = debt/income
    from scipy.special import expit

    # Noise
    noise = np.random.lognormal(mean=0, sigma=0.25, size=N)
    ltv = (ltv_base * noise).clip(0.20, 0.95).round(4)

    fico = fico - (60*ltv)
    fico = fico.round(0)

    # Normalize each feature to 0-1 scale
    fico_norm        = 1 - (fico - 300) / (850 - 300)        # invert — higher FICO = lower risk
    ltv_norm         = (ltv - 0.20) / (0.95 - 0.20)          # higher LTV = higher risk
    utilization_norm = utilization                             # already 0-1
    income_norm      = 1 - (income - 15_000) / (400_000 - 15_000)  # invert — higher income = lower risk

    # Weighted risk score — FICO carries the most weight
    risk_score = (
        0.40 * fico_norm +
        0.25 * ltv_norm +
        0.20 * utilization_norm +
        0.15 * income_norm
    )

    # Add small noise so ratings aren't perfectly deterministic
    noise = np.random.normal(0, 0.03, N)
    risk_score = (risk_score + noise).clip(0, 1)

    # Scale to 1-10
    risk_rating = (risk_score * 9 + 1).round(1)

    # Normalize drivers to 0-1
    fico_norm_lb     = (fico - 300) / (850 - 300)              # higher FICO = higher limit approved
    income_norm_lb   = (income - 15_000) / (400_000 - 15_000)  # higher income = higher balance
    debt_norm_lb     = 1 - (debt / debt.max())                  # lower existing debt = more headroom
    ltv_norm_lb      = 1 - (ltv - 0.20) / (0.95 - 0.20)       # lower LTV = more capacity
    util_norm_lb     = 1 - utilization                          # lower utilization = more capacity

    # Weighted combination
    loan_score = (
        0.30 * fico_norm_lb +
        0.30 * income_norm_lb +
        0.20 * debt_norm_lb +
        0.10 * ltv_norm_lb +
        0.10 * util_norm_lb
    )

    # Scale to 5k-75k
    loan_balance_base = 5_000 + loan_score * 70_000

    # Lognormal noise — loan sizing is behavioral and negotiated
    noise = np.random.lognormal(mean=0, sigma=0.50, size=N)
    loan_balance = (loan_balance_base * noise).clip(5_000, 75_000).round(2)

    # ── Probability of Default (logistic model) ──────────────────────────────

    # Normalize features for log-odds
    fico_effect        = -6.0 * (fico - 300) / (850 - 300)          # stronger FICO = much lower risk
    income_effect      = -4.5 * (income - 15_000) / (400_000 - 15_000)
    dti_effect         =  1.0 * dti.clip(0, 1)                       # DTI is a strong positive driver
    ltv_effect         =  2.0 * (ltv - 0.20) / (0.95 - 0.20)
    utilization_effect =  3.5 * utilization
    risk_rating_effect =  3.5 * (risk_rating - 1) / 9                          # 1-10 scale
    employment_effect  = -5.0 * employed                             # unemployed = much riskier
    age_effect         = -0.1 * (age - 25).clip(0)                 # older = slightly safer
    loan_effect        =  1.5 * (loan_balance - 5_000) / (75_000 - 5_000)

    log_odds = (
        -3.5       
        + fico_effect
        + income_effect
        + dti_effect
        + ltv_effect
        + utilization_effect
        + risk_rating_effect
        + employment_effect
        + age_effect
        + loan_effect
        + np.random.normal(0, 0.5, N)   # idiosyncratic noise
    )

    pd_prob = expit(log_odds)
    pd_prob += np.random.normal(0.0, 0.1, N)
    pd_prob = pd_prob.clip(0,1)
 
    default = np.random.binomial(1, pd_prob, N)

    # ── Delinquent Balance ────────────────────────────────────────────────────
    # Only nonzero for defaulted borrowers
    # Higher risk metrics = larger portion of loan balance is delinquent

    # Risk-based recovery rate — how much of the loan is lost
    fico_recovery      =  0.30 * (fico - 300) / (850 - 300)        # better FICO = more recovery
    utilization_loss   =  0.25 * utilization                        # maxed out = less recovery
    ltv_loss           =  0.20 * (ltv - 0.20) / (0.95 - 0.20)     # higher LTV = less recovery
    risk_rating_loss   =  0.15 * (risk_rating - 1) / 9             # higher rating = less recovery

    # Loss given default (LGD) — portion of loan balance that is delinquent
    lgd = (0.2 + utilization_loss + ltv_loss + risk_rating_loss - fico_recovery)
    lgd = (lgd + np.random.normal(0, 0.2, N)).clip(0.025, 0.975)

    # Delinquent balance only for defaulted borrowers
    delinquent_balance = np.where(default == 1, (loan_balance * lgd).round(2), 0.0)



    df = pd.DataFrame()

    df['Age'] = age
    df['Employed'] = employed
    df['Income'] = income

    df['Age and Income Interaction'] = age * income
    df['Employment and Income Interaction'] = employed * income
    df['Debt'] = debt
    df['Debt and Employment Interaction'] = debt*employed
    df['Debt to Income'] = debt/income
    df['Utilization Rate'] = utilization
    df['Utilization and Debt and Income Interaction'] = debt/income * utilization
    df['LTV'] = ltv
    df['Risk Rating'] = risk_rating
    df['Fico'] = fico
    df['Utilization and Risk Rating Interaction'] = utilization * risk_rating
    df['Utilization and Fico Interaction'] = utilization * fico
    df['LTV Fico Interaction'] = ltv*fico
    df['Risk Rating Fico Interaction'] = risk_rating * fico
    df['Fico and Income Interaction'] = fico*income
    df['Balance'] = loan_balance
    df['Balance and Risk Rating Interaction'] = loan_balance * risk_rating
    df['Balance and FICO Interaction'] = loan_balance * fico
    df['Balance and Debt Interaction'] = loan_balance * debt
    df['Default'] = default


    loan_id = 1

    ids = []

    for x in range(len(delinquent_balance)):

        ids.append(loan_id)
        loan_id +=1

    df['Loan ID'] = ids


    #df.to_csv(complete_dataset_path, index=False)

    engine_path = "sqlite:///" + db_name + ".db"

    engine = create_engine(engine_path)

    #df = pd.read_csv(complete_dataset_path, index_col=False)

    df.to_sql("CreditRiskCompleteData", engine, if_exists="replace", index=False)

    df = df.drop(['Loan ID','Default'], axis=1)

    df.to_sql("CreditRiskTrainingData", engine, if_exists="replace", index=False)
    

    print(f"Done! {len(df)} rows inserted.")

    df = pd.read_sql("SELECT * FROM CreditRiskTrainingData", engine)
    print(df)
