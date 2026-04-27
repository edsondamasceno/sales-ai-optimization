import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    df['order_date'] = pd.to_datetime(df['order_date'])

    return df