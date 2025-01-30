import pandas as pd
import streamlit as st
from PIL import Image
import sqlite3
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Helper import *

# ============= PAGE SETUP ============ 
st.set_page_config(page_title="Employment Analytics", page_icon="📊", layout="wide")

# =========== Initialize SQL ========== 
conn = sqlite3.connect("Data.db")
cursor_object = conn.cursor()

# Fetch the data from the database
query = "SELECT * FROM COVID19"
data = pd.read_sql_query(query, conn)
data['Date'] = pd.to_datetime(data['Date'])

# ============= PAGE HEADER ============ 
c1, c2 = st.columns([7, 3])
with c1:
    st.subheader("Employment Analytics Interactive Dashboard")
with c2:
    years = sorted(list(data['Date'].dt.year.unique()), reverse=True)
    years.insert(0, "All")
    year = st.selectbox('Filter by year', options=years, index=0)

if year != "All":
    data = data[data['Date'].dt.year == int(year)]

# ================== Row 1: Key Metrics ===================== 
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_employed = data['Total_employed_persons'].iloc[-1]
    plot_metric(label="Total Employed (K)", value=total_employed)

with col2:
    unemployment_rate = data['unemployement_rate'].iloc[-1]
    plot_metric(label="Unemployment Rate (%)", value=unemployment_rate)

with col3:
    participation_rate = data['Labor_force_participation_rate'].iloc[-1]
    plot_metric(label="Labor Force Participation (%)", value=participation_rate)

with col4:
    employment_rate = data['employment_rate'].iloc[-1]
    plot_metric(label="Employment Rate (%)", value=employment_rate)

with col5:
    youth_unemployment = data['Unemployment_rate_age_15-30'].iloc[-1]
    plot_metric(label="Youth Unemployment (%)", value=youth_unemployment)

with col6:
    outside_labor = data['Outside_Labor_force'].iloc[-1]
    plot_metric(label="Outside Labor Force (K)", value=outside_labor)

st.write('---')

# ================== Employment Trends ===================== 
col1, col2 = st.columns(2)
with col1:
    st.caption("**Total Employment Over Time**")
    employment_trend(data)
with col2:
    st.caption("**Unemployment Rate Trends**")
    unemployment_trend(data)

st.write('---')

# ================== Age Group Analysis ===================== 
col1, col2 = st.columns(2)
with col1:
    st.caption("**Underemployment Rates by Age Group**")
    age_group_analysis(data)
with col2:
    st.caption("**Labor Force Participation Trends**")
    participation_trend(data)

st.write('---')

# ================== Monthly Changes ===================== 
st.caption("**Monthly Employment Changes**")
monthly_changes(data)

st.write('---')

# ================== Summary: Key Labour Market Impacts =====================
st.markdown("### Summary of Key Labour Market Impacts During the Pandemic")

col1, col2 = st.columns(2)

with col1:
    st.caption("**Unemployment Rate Over Time**")
    fig_unemployment = px.line(data, x='Date', y='unemployement_rate',
                                labels={'unemployement_rate': 'Unemployment Rate (%)', 'Date': 'Date'},
                                color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_unemployment, use_container_width=True)

with col2:
    st.caption("**Persons Outside Labor Force Over Time**")
    fig_outside_labor = px.bar(data, x='Date', y='Outside_Labor_force',
                               labels={'Outside_Labor_force': 'Persons Outside Labor Force (Thousands)', 'Date': 'Date'},
                               color_discrete_sequence=['#EF553B'])
    st.plotly_chart(fig_outside_labor, use_container_width=True)

st.write('---')

# ================== Employment Rate Prediction ===================== 
st.subheader("Employment Rate Prediction")

# Data Preprocessing
df = data.copy()
df = df.drop(columns=['Date'])
df.columns = df.columns.str.strip()

X = df.drop(columns=['employment_rate'])
y = df['employment_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R²", f"{r2:.2f}")

# Plot Actual vs Predicted Employment Rate
pred_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Date': data['Date'].iloc[-len(y_test):]  # Use last n dates for visualization
})

fig = px.line(pred_df, x='Date', y=['Actual', 'Predicted'],
              labels={'value': 'Employment Rate (%)', 'Date': 'Date'},
              title="Actual vs Predicted Employment Rate",
              color_discrete_map={'Actual': 'blue', 'Predicted': 'red'})

st.plotly_chart(fig, use_container_width=True)
