import pandas as pd

def prepare_features(df, window=14):

    df_daily = df.groupby(['order_date', 'sub_category'])['quantity'].sum().reset_index()
    df_daily = df_daily.sort_values('order_date')

    features = []

    for lag in range(1, window + 1):
        col = f'lag_{lag}'
        df_daily[col] = df_daily.groupby('sub_category')['quantity'].shift(lag)
        features.append(col)

    df_daily['day_of_week'] = df_daily['order_date'].dt.dayofweek
    df_daily['month'] = df_daily['order_date'].dt.month

    df_daily = df_daily.dropna()

    return df_daily, features