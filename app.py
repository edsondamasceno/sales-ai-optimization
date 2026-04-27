import streamlit as st
import pandas as pd
import plotly.express as px

from core.features import prepare_features
from core.forecasting import train_models, predict_n_days
from core.optimization import optimize_price_stock
from models.train import load_data
from config import WINDOW, HORIZON

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Revenue Engine", layout="wide")

st.title("💰 AI Revenue Optimization Engine")

# =========================
# LOAD DATA
# =========================
df = load_data("data/Sales-Dataset.csv")
df_features, feature_cols = prepare_features(df, WINDOW)
models = train_models(df_features, feature_cols, HORIZON)

products = list(models.keys())

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")

selected_product = st.sidebar.selectbox("Select Product", products)

stock = st.sidebar.slider("Stock", 0, 500, 100)
base_price = st.sidebar.slider("Base Price", 10, 200, 50)
cost = st.sidebar.slider("Cost", 5, 100, 20)

simulate_price = st.sidebar.slider("Simulate Price", 10, 200, base_price)

# =========================
# GET DATA
# =========================
model = models[selected_product]

preds = predict_n_days(
    df_features,
    model,
    selected_product,
    feature_cols,
    HORIZON
)

opt_price, opt_order, opt_profit = optimize_price_stock(
    preds, stock, base_price, cost
)

# Simulação manual
simulated_demand = [p * (simulate_price / base_price) ** -1.3 for p in preds]
simulated_profit = sum(simulated_demand) * simulate_price

current_profit = sum(preds) * base_price

# =========================
# KPIs
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Current Profit", f"R$ {int(current_profit):,}")
col2.metric("🚀 Optimized Profit", f"R$ {int(opt_profit):,}")
col3.metric("📈 Optimal Price", f"R$ {opt_price:.2f}")
col4.metric("📦 Optimal Order", int(opt_order))

# =========================
# DEMAND FORECAST
# =========================
st.subheader(f"📈 Demand Forecast - {selected_product}")

df_pred = pd.DataFrame({
    "Day": list(range(1, HORIZON + 1)),
    "Forecast": preds,
    "Simulated": simulated_demand
})

fig = px.line(
    df_pred,
    x="Day",
    y=["Forecast", "Simulated"],
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# PROFIT COMPARISON
# =========================
st.subheader("💰 Profit Scenarios")

df_profit = pd.DataFrame({
    "Scenario": ["Current", "Optimized", "Simulated"],
    "Profit": [current_profit, opt_profit, simulated_profit]
})

fig2 = px.bar(df_profit, x="Scenario", y="Profit")

st.plotly_chart(fig2, use_container_width=True)

# =========================
# TABLE ALL PRODUCTS
# =========================
st.subheader("📊 All Products Overview")

rows = []

for product, model in models.items():

    preds = predict_n_days(df_features, model, product, feature_cols, HORIZON)

    price, order, profit = optimize_price_stock(
        preds, stock, base_price, cost
    )

    rows.append({
        "Product": product,
        "Optimal Price": round(price, 2),
        "Optimal Order": order,
        "Max Profit": int(profit)
    })

df_all = pd.DataFrame(rows).sort_values(by="Max Profit", ascending=False)

st.dataframe(df_all, use_container_width=True)