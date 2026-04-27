# =========================================
# SALES AI - MODERN DASHBOARD (STRIPE STYLE)
# =========================================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Sales AI", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Sales-Dataset.csv")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    df['order_date'] = pd.to_datetime(df['order_date'])

    return df

# =========================
# FEATURES
# =========================
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

# =========================
# MODEL
# =========================
def train_all_models(df, features, horizon=7):

    models, maes = {}, {}

    for product in df['sub_category'].unique():
        data = df[df['sub_category'] == product].copy()

        if len(data) < 30:
            continue

        data['target'] = data['quantity'].shift(-horizon)
        data = data.dropna()

        X = data[features + ['day_of_week', 'month']]
        y = data['target']

        split = int(len(X) * 0.8)

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X[:split], y[:split])

        preds = model.predict(X[split:])
        maes[product] = mean_absolute_error(y[split:], preds)

        models[product] = model

    return models, maes

# =========================
# FORECAST
# =========================
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

# =========================
# STOCK
# =========================
def predict_stockout_day(preds, stock):
    cumulative = np.cumsum(preds)
    for i, val in enumerate(cumulative):
        if val >= stock:
            return i + 1
    return None

# =========================
# UI HEADER
# =========================
st.markdown("""
    <h1 style='font-size:32px;'>📊 Sales AI Dashboard</h1>
    <p style='color:gray;'>Real-time demand forecasting & inventory intelligence</p>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    horizon = st.slider("Forecast Horizon", 7, 30, 7)
    stock = st.number_input("Stock Level", value=100)
    lead_time = st.slider("Lead Time", 1, 15, 5)
    load_btn = st.button("Load Data")

# =========================
# MAIN
# =========================
if load_btn:
    st.session_state.df = load_data()

if 'df' in st.session_state:

    df = st.session_state.df

    df_features, feature_cols = prepare_features(df)

    # KPIs (cards style)
    total_sales = df['amount'].sum()
    total_orders = df['order_id'].nunique()
    total_customers = df['customername'].nunique()

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
        <div style='padding:20px;border-radius:12px;background:#111;'>
        <h3>💰 Revenue</h3>
        <h2>{total_sales:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
        <div style='padding:20px;border-radius:12px;background:#111;'>
        <h3>🧾 Orders</h3>
        <h2>{total_orders}</h2>
        </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
        <div style='padding:20px;border-radius:12px;background:#111;'>
        <h3>👥 Customers</h3>
        <h2>{total_customers}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🚀 Train Models"):
        models, maes = train_all_models(df_features, feature_cols, horizon)
        st.session_state.models = models
        st.session_state.maes = maes

    if 'models' in st.session_state:

        rows = []

        for product, model in st.session_state.models.items():

            preds = predict_n_days(df_features, model, product, feature_cols, horizon)
            stockout = predict_stockout_day(preds, stock)

            status = "🟢 OK"
            if stockout:
                status = "🔴 URGENT" if stockout <= lead_time else "🟡 RISK"

            rows.append({
                'Product': product,
                'Forecast': round(preds[0], 2),
                'Stockout (days)': stockout,
                'Status': status,
                'Error': round(st.session_state.maes[product], 2)
            })

        dashboard = pd.DataFrame(rows)

        st.subheader("📋 Inventory Overview")
        st.dataframe(dashboard, use_container_width=True)

        colA, colB = st.columns(2)

        with colA:
            st.subheader("📈 Status Distribution")
            st.bar_chart(dashboard['Status'].value_counts())

        with colB:
            st.subheader("🔥 Critical Products")
            st.dataframe(dashboard.sort_values(by='Stockout (days)').head(10))

else:
    st.info("Load data to start")

# =========================================
# END
# =========================================
