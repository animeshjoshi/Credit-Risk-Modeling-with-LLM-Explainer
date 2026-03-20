import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData
import Simulate
import ModelRisk
import Transformations


def build_tabular_db(db_name, sample_size):

    engine_path = "sqlite:///" + db_name + ".db"
    engine = create_engine(engine_path)
    meta = MetaData()
    meta.reflect(bind=engine)
    meta.drop_all(bind=engine)




    Simulate.simulate_risk_data(sample_size, db_name)


    X_data = pd.read_sql("SELECT * FROM CreditRiskTrainingData", engine)



    y_data = pd.read_sql("SELECT * FROM CreditRiskCompleteData", engine)['Default']


    ModelRisk.logistic_pd(X=X_data.copy(), y=y_data.copy(), db_path = db_name, model_identifier="Logistic Regression PD Model on Raw Loan Data", reg=False)

    X_woe = Transformations.weight_of_evidence(X_data.copy(), y_data.copy()).drop('Employed', axis = 1)

    ModelRisk.logistic_pd(X=X_woe.copy(), y=y_data.copy(), db_path = db_name, model_identifier="Logistic Regression PD Model on Weight of Evidence Loan Data", reg=True)



    ModelRisk.xgboost_pd(X=X_data.copy(), y=y_data.copy(), db_path = db_name, model_identifier="XGBoost PD Model on Raw Loan Data")

