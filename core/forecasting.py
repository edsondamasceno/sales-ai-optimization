import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_models(df, features, horizon=7):

    models = {}

    for product in df['sub_category'].unique():

        data = df[df['sub_category'] == product].copy()

        if len(data) < 30:
            continue

        data['target'] = data['quantity'].shift(-horizon)
        data = data.dropna()

        X = data[features + ['day_of_week', 'month']]
        y = data['target']

        model = RandomForestRegressor(n_estimators=150)
        model.fit(X, y)

        models[product] = model

    return models


def predict_n_days(df, model, product, features, n_days=7):

    data = df[df['sub_category'] == product].copy()
    last_row = data.iloc[-1].copy()

    preds = []

    for _ in range(n_days):

        X = pd.DataFrame([{
            **{f: last_row[f] for f in features},
            'day_of_week': last_row['day_of_week'],
            'month': last_row['month']
        }])

        pred = model.predict(X)[0]
        preds.append(pred)

        for i in range(len(features), 1, -1):
            last_row[f'lag_{i}'] = last_row[f'lag_{i-1}']

        last_row['lag_1'] = pred
        last_row['quantity'] = pred
        last_row['day_of_week'] = (last_row['day_of_week'] + 1) % 7

    return preds