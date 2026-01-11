import pandas as pd

def preprocess(path):
    df = pd.read_csv(path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    df['churn'] = (df['churn'] == 'yes').astype(int)

    binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
    for col in binary_cols:
        df[col] = (df[col] == 'yes').astype(int)

    internet_cols = [
        'onlinesecurity','onlinebackup','deviceprotection',
        'techsupport','streamingtv','streamingmovies','multiplelines'
    ]
    for col in internet_cols:
        df[col] = df[col].replace({
            'no_internet_service': 'no',
            'no_phone_service': 'no'
        })
        df[col] = (df[col] == 'yes').astype(int)

    df['gender'] = (df['gender'] == 'male').astype(int)

    df['contract'] = df['contract'].map({
        'month-to-month': 0,
        'one_year': 1,
        'two_year': 2
    }).astype(int)

    df = pd.get_dummies(df, columns=['paymentmethod'], prefix='payment', drop_first=True)
    df = pd.get_dummies(df, columns=['internetservice'], prefix='service', drop_first=True)

    df = df.drop('customerid', axis=1)

    return df
