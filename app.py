import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Page Configuration
st.set_page_config(page_title="NCRB Crime Intelligence Hub", layout="wide")

# 1. DATA ENGINE
@st.cache_data
def load_ncrb_data():
    # Loading the dataset provided in the repository
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_ncrb_data()

# --- SIDEBAR FILTERS ---
st.sidebar.title("🔍 Dashboard Controls")
st.sidebar.markdown("Filter the historical data and configure the AI.")
selected_year = st.sidebar.slider("Historical Data Range", 2001, 2013, (2001, 2013))
selected_states = st.sidebar.multiselect("Filter by State/UT", sorted(df['STATE/UT'].unique()))

# Filter logic for charts
filtered_df = df[(df['YEAR'] >= selected_year[0]) & (df['YEAR'] <= selected_year[1])]
if selected_states:
    filtered_df = filtered_df[filtered_df['STATE/UT'].isin(selected_states)]

# --- I. KEY ESSENTIALS: KPI CARDS ---
st.title("🛡️ Indian Crime Intelligence & AI Forecast")
st.markdown("### National Crime Records Bureau (NCRB) Analysis Tool")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_ipc = filtered_df['TOTAL IPC CRIMES'].sum()
avg_crime = filtered_df['TOTAL IPC CRIMES'].mean()
top_state = filtered_df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().idxmax()

kpi1.metric("Total IPC Cases", f"{total_ipc:,}")
kpi2.metric("Avg. Yearly Crimes", f"{int(avg_crime):,}")
kpi3.metric("Highest Volume State", top_state)
kpi4.metric("Analysis Period", f"{selected_year[0]}-{selected_year[1]}")

st.markdown("---")

# --- II. GEOSPATIAL & DISTRIBUTION ---
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader("📍 State-wise Crime Intensity")
    # Interactive bar chart acting as a heat intensity map
    state_totals = filtered_df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index()
    fig_map = px.bar(state_totals, x='STATE/UT', y='TOTAL IPC CRIMES', 
                     color='TOTAL IPC CRIMES', color_continuous_scale='Reds',
                     labels={'TOTAL IPC CRIMES':'Total Cases'},
                     title="Crime Volume by Geography")
    st.plotly_chart(fig_map, use_container_width=True)

with row1_col2:
    st.subheader("📊 Crime Category Breakdown")
    crime_types = ['MURDER', 'ROBBERY', 'BURGLARY', 'THEFT', 'KIDNAPPING & ABDUCTION']
    crime_sums = filtered_df[crime_types].sum().reset_index()
    crime_sums.columns = ['Crime', 'Count']
    fig_donut = px.pie(crime_sums, values='Count', names='Crime', hole=0.5, 
                       color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_donut, use_container_width=True)

# --- III. TIME SERIES & PREDICTIVE ---
st.markdown("---")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📈 Historical Crime Trend")
    trend_data = filtered_df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().reset_index()
    fig_trend = px.line(trend_data, x='YEAR', y='TOTAL IPC CRIMES', markers=True, 
                        line_shape='spline', title="Yearly Fluctuation (2001-2013)")
    st.plotly_chart(fig_trend, use_container_width=True)

with row2_col2:
    st.subheader("🔮 AI Deep Learning Forecast (2027-2035)")
    target_year = st.slider("Select Future Year to Predict", 2027, 2035, 2030)
    
    if st.button("🚀 Generate Neural Prediction"):
        # ANN Forecasting Logic
        X = df[['YEAR']].values
        y = df['TOTAL IPC CRIMES'].values
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Artificial Neural Network Architecture
        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(32, activation='relu'),
            Dense(1) 
        ])
        model.compile(optimizer='adam', loss='mse')
        
        with st.spinner('Neural Network is training on historical patterns...'):
            model.fit(X_scaled, y_scaled, epochs=100, verbose=0)
        
        # Prediction for target year
        input_year = np.array([[target_year]])
        input_scaled = scaler_X.transform(input_year)
        prediction_scaled = model.predict(input_scaled)
        final_prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        # Results and Managerial Implications
        st.metric(f"Predicted Total IPC for {target_year}", f"{int(final_prediction):,}")
        
        baseline_2013 = df[df['YEAR'] == 2013]['TOTAL IPC CRIMES'].sum()
        pct_change = ((final_prediction - baseline_2013) / baseline_2013) * 100
        
        if pct_change > 0:
            st.error(f"⚠️ Trend: {pct_change:.1f}% Increase predicted compared to 2013.")
            st.info("Managerial Recommendation: Scale up law enforcement resources.")
        else:
            st.success(f"📉 Trend: {abs(pct_change):.1f}% Decrease predicted compared to 2013.")
            st.info("Managerial Recommendation: Maintain current crime prevention strategies.")

# --- IV. STATE COMPARISON ---
st.markdown("---")
st.subheader("⚖️ Comparative Analysis Engine")
comp_states = st.multiselect("Select Two States for Direct Comparison", 
                             sorted(df['STATE/UT'].unique()), default=sorted(df['STATE/UT'].unique())[:2])

if len(comp_states) >= 2:
    comparison_df = df[df['STATE/UT'].isin(comp_states)]
    fig_comp = px.bar(comparison_df, x='YEAR', y='TOTAL IPC CRIMES', 
                      color='STATE/UT', barmode='group', title="State Comparison Over Time")
    st.plotly_chart(fig_comp, use_container_width=True)
