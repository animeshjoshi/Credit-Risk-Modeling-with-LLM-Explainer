from sqlalchemy import create_engine
import pandas as pd

# Update these values

engine = create_engine("sqlite:///Credit_Risk_DB.db")

df = pd.read_csv("Credit Risk.csv", index_col=False)
print(df)
df = df.drop(['Default', 'Delinquent Balance'], axis=1)
df.to_sql("my_table", engine, if_exists="replace", index=False)
df.to_csv("Credit Risk (Training Data).csv")

print(f"Done! {len(df)} rows inserted.")

df = pd.read_sql("SELECT * FROM my_table", engine)
print(df)