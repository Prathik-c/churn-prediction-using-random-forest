import joblib
import pandas as pd
from preprocessing import preprocess

print("Loading model...")
model = joblib.load("churn_model.pkl")

print("Reading & preprocessing data...")
df_new = preprocess("Customer-Churn.CSV")

print("Data shape after preprocessing:", df_new.shape)
print("Columns:", df_new.columns)

if 'churn' in df_new.columns:
    df_new = df_new.drop(columns=['churn'])

print("Running predictions...")
pred = model.predict(df_new)
proba = model.predict_proba(df_new)[:, 1]

df_new['churn_prediction'] = pred
df_new['churn_probability'] = proba

print("\nSample predictions:")
print(df_new[['churn_prediction', 'churn_probability']].head(10))

