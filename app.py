import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Page Configuration
st.set_page_config(page_title="NCRB AI Intelligence Hub", layout="wide")

# 1. DATA ENGINE
@st.cache_data
def load_ncrb_data():
    df = pd.read_csv('Crimes_in_india_2001-2013.csv')
    df.columns = df.columns.str.strip()
    return df

df = load_ncrb_data()
crime_list = ['TOTAL IPC CRIMES', 'MURDER', 'ROBBERY', 'BURGLARY', 'THEFT', 'RAPE', 'KIDNAPPING & ABDUCTION']

# --- SIDEBAR: MULTILEVEL FILTERS (Requirement II) ---
st.sidebar.title("🔍 Strategic Filters")
target_crime = st.sidebar.selectbox("🎯 Select Primary Crime Category", crime_list)
selected_year = st.sidebar.slider("Historical Range", 2001, 2013, (2001, 2013))
selected_states = st.sidebar.multiselect("Focus States", sorted(df['STATE/UT'].unique()))

filtered_df = df[(df['YEAR'] >= selected_year[0]) & (df['YEAR'] <= selected_year[1])]
if selected_states:
    filtered_df = filtered_df[filtered_df['STATE/UT'].isin(selected_states)]

# --- I. KPI CARDS (Requirement I) ---
st.title("🛡️ Indian Crime Intelligence & ANN Forecast")
st.markdown(f"### Analysis Target: **{target_crime}**")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(f"Total {target_crime}", f"{filtered_df[target_crime].sum():,}")
kpi2.metric("Annual Average", f"{int(filtered_df[target_crime].mean()):,}")
kpi3.metric("Highest Intensity", filtered_df.groupby('STATE/UT')[target_crime].sum().idxmax())
kpi4.metric("YoY Variance", f"{((df.groupby('YEAR')[target_crime].sum().pct_change().iloc[-1])*100):.2f}%")

st.markdown("---")

# --- II. ADVANCED VISUALS (Requirement I & II) ---
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader("📦 Statistical Distribution & Outliers")
    # New Box Plot for deep statistical insight
    fig_box = px.box(filtered_df, x='YEAR', y=target_crime, points="all", 
                     color_discrete_sequence=['#d62728'], title=f"Spread of {target_crime} across States")
    st.plotly_chart(fig_box, use_container_width=True)

with row1_col2:
    st.subheader("🍩 State Contribution")
    # Donut showing top 5 contributors
    top_5 = filtered_df.groupby('STATE/UT')[target_crime].sum().nlargest(5).reset_index()
    fig_donut = px.pie(top_5, values=target_crime, names='STATE/UT', hole=0.5, 
                       color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_donut, use_container_width=True)

# --- III. TRENDS & GROWTH ---
st.markdown("---")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📈 Yearly Growth Velocity (%)")
    # New Bar chart for growth rates
    growth_data = df.groupby('YEAR')[target_crime].sum().pct_change() * 100
    fig_growth = px.bar(growth_data, x=growth_data.index, y=target_crime, 
                        labels={target_crime: 'Growth %'}, color=target_crime,
                        color_continuous_scale='RdYlGn_r', title="Momentum Analysis")
    st.plotly_chart(fig_growth, use_container_width=True)

with row2_col2:
    st.subheader("🔮 AI Neural Forecast (to 2035)")
    target_year = st.slider("Select Forecast Horizon", 2026, 2035, 2030)
    
    if st.button("🚀 Execute ANN Forecast"):
        X = df[['YEAR']].values
        y = df[target_crime].values
        
        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_s, y_s = sc_X.fit_transform(X), sc_y.fit_transform(y.reshape(-1, 1))
        
        # ANN Architecture (Requirement C)
        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        with st.spinner('AI analyzing historical cycles...'):
            model.fit(X_s, y_s, epochs=100, verbose=0)
        
        pred = sc_y.inverse_transform(model.predict(sc_X.transform([[target_year]])))[0][0]
        st.metric(f"AI Prediction for {target_year}", f"{int(pred):,}")
        
        # Trend comparison
        baseline = df[df['YEAR'] == 2013][target_crime].sum()
        change = ((pred - baseline) / baseline) * 100
        st.info(f"The ANN predicts a **{change:+.2f}%** change from the 2013 baseline.")

# --- IV. COMPARISON ENGINE ---
st.markdown("---")
st.subheader("⚖️ Side-by-Side Comparison Engine")
comp_states = st.multiselect("Choose States to Compare", sorted(df['STATE/UT'].unique()), default=sorted(df['STATE/UT'].unique())[:2])

if len(comp_states) >= 2:
    comp_df = df[df['STATE/UT'].isin(comp_states)]
    fig_comp = px.area(comp_df, x='YEAR', y=target_crime, color='STATE/UT', 
                       title=f"Volume Comparison for {target_crime}")
    st.plotly_chart(fig_comp, use_container_width=True)
