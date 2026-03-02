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
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_ncrb_data()

# --- SIDEBAR FILTERS (Requirement II) ---
st.sidebar.title("🔍 Advanced Filters")
# Interactive Slicer for Crime Categories
crime_options = ['TOTAL IPC CRIMES', 'MURDER', 'ROBBERY', 'BURGLARY', 'THEFT', 'RAPE', 'KIDNAPPING & ABDUCTION']
selected_crime = st.sidebar.selectbox("🎯 Select Crime Category to Analyze", crime_options)

selected_year = st.sidebar.slider("Historical Data Range", 2001, 2013, (2001, 2013))
selected_states = st.sidebar.multiselect("Filter by State/UT", sorted(df['STATE/UT'].unique()))

# Filter logic
filtered_df = df[(df['YEAR'] >= selected_year[0]) & (df['YEAR'] <= selected_year[1])]
if selected_states:
    filtered_df = filtered_df[filtered_df['STATE/UT'].isin(selected_states)]

# --- I. KPI CARDS ---
st.title("🛡️ Indian Crime Intelligence & AI Forecast")
st.markdown(f"### Analyzing: {selected_crime}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_val = filtered_df[selected_crime].sum()
avg_val = filtered_df[selected_crime].mean()
top_state = filtered_df.groupby('STATE/UT')[selected_crime].sum().idxmax()

kpi1.metric(f"Total {selected_crime}", f"{total_val:,}")
kpi2.metric("Avg Annual Volume", f"{int(avg_val):,}")
kpi3.metric("Highest Volume State", top_state)
kpi4.metric("Active Filters", f"{len(filtered_df['STATE/UT'].unique())} States")

st.markdown("---")

# --- II. GEOSPATIAL & CONTRIBUTION DONUT ---
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader(f"📍 Regional Intensity: {selected_crime}")
    state_totals = filtered_df.groupby('STATE/UT')[selected_crime].sum().reset_index()
    fig_bar = px.bar(state_totals, x='STATE/UT', y=selected_crime, 
                     color=selected_crime, color_continuous_scale='Reds',
                     title=f"Geographic Distribution of {selected_crime}")
    st.plotly_chart(fig_bar, use_container_width=True)

with row1_col2:
    st.subheader("🍩 State Contribution (%)")
    # Show top 5 states and group others for a clean donut
    top_5 = filtered_df.groupby('STATE/UT')[selected_crime].sum().sort_values(ascending=False).head(5).reset_index()
    fig_donut = px.pie(top_5, values=selected_crime, names='STATE/UT', hole=0.6, 
                       color_discrete_sequence=px.colors.sequential.RdBu)
    fig_donut.update_layout(showlegend=False) # Keep it clean
    st.plotly_chart(fig_donut, use_container_width=True)

# --- III. TIME SERIES & AI FORECAST ---
st.markdown("---")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📈 Historical Trend Line")
    trend_data = filtered_df.groupby('YEAR')[selected_crime].sum().reset_index()
    fig_trend = px.line(trend_data, x='YEAR', y=selected_crime, markers=True, 
                        title=f"Yearly Trend of {selected_crime} (2001-2013)")
    st.plotly_chart(fig_trend, use_container_width=True)

with row2_col2:
    st.subheader(f"🔮 AI Forecast: {selected_crime} (to 2035)")
    target_year = st.slider("Select Forecast Year", 2026, 2035, 2030)
    
    if st.button("🚀 Run Neural Forecast"):
        # Training Logic
        X = df[['YEAR']].values
        y = df[selected_crime].values
        
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_s, y_s = scaler_X.fit_transform(X), scaler_y.fit_transform(y.reshape(-1, 1))
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        with st.spinner('Neural Network training...'):
            model.fit(X_s, y_s, epochs=100, verbose=0)
        
        pred = scaler_y.inverse_transform(model.predict(scaler_X.transform([[target_year]])))[0][0]
        st.metric(f"Predicted {selected_crime} (Year {target_year})", f"{int(pred):,}")
        
        # Comparison logic
        baseline = df[df['YEAR'] == 2013][selected_crime].sum()
        diff = ((pred - baseline) / baseline) * 100
        st.write(f"Trend: **{diff:+.1f}%** relative to 2013 baseline.")

# --- IV. DATA INSPECTOR & COMPARISON ---
st.markdown("---")
tab1, tab2 = st.tabs(["⚖️ State Comparison Engine", "🗂️ Detailed Data Inspector"])

with tab1:
    comp_states = st.multiselect("Select States for Side-by-Side Analysis", 
                                 sorted(df['STATE/UT'].unique()), default=sorted(df['STATE/UT'].unique())[:2])
    if len(comp_states) >= 2:
        comp_df = df[df['STATE/UT'].isin(comp_states)]
        fig_comp = px.bar(comp_df, x='YEAR', y=selected_crime, color='STATE/UT', barmode='group')
        st.plotly_chart(fig_comp, use_container_width=True)

with tab2:
    st.subheader("Reference Data Table")
    st.dataframe(filtered_df[['STATE/UT', 'YEAR'] + crime_options], use_container_width=True)
