import pandas as pd
import numpy as np
import Simulate
import ModelRisk


Simulate.simulate_risk_data(100000, "Data/Credit Risk Full Dataset.csv", "Data/Credit Risk Training Dataset.csv", "CredRiskDB")
X_data = pd.read_csv('Data/Credit Risk Training Dataset.csv').drop('Unnamed: 0', axis = 1)
y_data = pd.read_csv('Data/Credit Risk Full Dataset.csv')['Default']
new_data = ModelRisk.logistic_pd(X_data, y_data, "Data/Logistic PD Summary.csv", "CredRiskDB")
print(new_data)
