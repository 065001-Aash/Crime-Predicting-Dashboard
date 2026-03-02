import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Page Configuration
st.set_page_config(page_title="NCRB Crime Analytics Hub", layout="wide")

# 1. DATA ENGINE (Optimized)
@st.cache_data
def load_ncrb_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_ncrb_data()

# --- SIDEBAR FILTERS (Requirement II) ---
st.sidebar.title("🔍 Deep-Dive Filters")
selected_year = st.sidebar.slider("Select Year Range", 2001, 2013, (2001, 2013))
selected_states = st.sidebar.multiselect("Filter by State/UT", sorted(df['STATE/UT'].unique()))

# Filter logic
filtered_df = df[(df['YEAR'] >= selected_year[0]) & (df['YEAR'] <= selected_year[1])]
if selected_states:
    filtered_df = filtered_df[filtered_df['STATE/UT'].isin(selected_states)]

# --- I. KEY ESSENTIALS: KPI CARDS (Requirement I) ---
st.title("🛡️ Indian Crime Intelligence Dashboard")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_ipc = filtered_df['TOTAL IPC CRIMES'].sum()
avg_crime = filtered_df['TOTAL IPC CRIMES'].mean()
top_state = filtered_df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().idxmax()

kpi1.metric("Total IPC Cases", f"{total_ipc:,}")
kpi2.metric("Avg. Crime Volume", f"{int(avg_crime):,}")
kpi3.metric("Highest Crime State", top_state)
kpi4.metric("Data Horizon", f"{selected_year[0]}-{selected_year[1]}")

st.markdown("---")

# --- II. GEOSPATIAL & DISTRIBUTION ---
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader("📍 Crime Density Heatmap (State-wise)")
    # Note: For a true choropleth, you'd link to an India GeoJSON. 
    # Using a bar chart as a data proxy for now.
    fig_map = px.bar(filtered_df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index(), 
                     x='STATE/UT', y='TOTAL IPC CRIMES', color='TOTAL IPC CRIMES',
                     color_continuous_scale='Reds', title="Crime Intensity Index")
    st.plotly_chart(fig_map, use_container_width=True)

with row1_col2:
    st.subheader("📊 Crime Type Breakdown")
    crime_types = ['MURDER', 'ATTEMPT TO MURDER', 'ROBBERY', 'BURGLARY', 'THEFT']
    crime_sums = filtered_df[crime_types].sum().reset_index()
    crime_sums.columns = ['Crime', 'Count']
    fig_donut = px.pie(crime_sums, values='Count', names='Crime', hole=0.5, 
                       color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_donut, use_container_width=True)

# --- III. TIME SERIES & PREDICTIVE (Requirement II) ---
st.markdown("---")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📈 Historical Time-Series Trend")
    trend_data = filtered_df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().reset_index()
    fig_trend = px.line(trend_data, x='YEAR', y='TOTAL IPC CRIMES', markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

with row2_col2:
    st.subheader("🔮 AI Future Forecast (2026-2035)")
    # Simple ANN for prediction
    if st.button("Generate Neural Forecast"):
        # Training logic (Simplified for dashboard speed)
        X = df[['YEAR']].values
        y = df['TOTAL IPC CRIMES'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = Sequential([Dense(10, activation='relu', input_shape=(1,)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y, epochs=50, verbose=0)
        
        # Forecast years
        future_years = np.array(range(2026, 2036)).reshape(-1, 1)
        future_scaled = scaler.transform(future_years)
        preds = model.predict(future_scaled)
        
        forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Crime': preds.flatten()})
        st.line_chart(forecast_df, x='Year', y='Predicted Crime')
        st.success("Forecast generated using Sequential Artificial Neural Network.")

# --- IV. COMPARISON TOGGLE (Requirement II) ---
st.markdown("---")
st.subheader("⚖️ State Comparison Engine")
comp_states = st.multiselect("Select Two States for Side-by-Side Comparison", 
                             sorted(df['STATE/UT'].unique()), default=sorted(df['STATE/UT'].unique())[:2])

if len(comp_states) >= 2:
    comparison_df = df[df['STATE/UT'].isin(comp_states)]
    fig_comp = px.bar(comparison_df, x='YEAR', y='TOTAL IPC CRIMES', color='STATE/UT', barmode='group')
    st.plotly_chart(fig_comp, use_container_width=True)
